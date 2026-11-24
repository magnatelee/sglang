from __future__ import annotations

import logging
import time
import types
import weakref
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import torch

from sglang.srt.utils import get_device_module, is_hip, is_xpu

if is_xpu():
    from sgl_kernel import (
        load_cache_to_device_buffer_dsv4_mla,
        load_cache_to_device_buffer_mla,
    )

    def copy_cache_planned_mla(*args, **kwargs):
        raise RuntimeError(
            "HiSparse shared-index prefetch is unsupported on XPU: "
            "copy_cache_planned_mla has no AOT sgl_kernel implementation."
        )

    def load_blocks_to_device_buffer_mha(*args, **kwargs):
        raise RuntimeError(
            "MiniMax M3 HiSparse block swap-in is unsupported on XPU: "
            "load_blocks_to_device_buffer_mha has no AOT sgl_kernel implementation."
        )

else:
    from sglang.kernels.ops.kvcache.hisparse import (
        HiSparseSpecState,
        copy_cache_planned_mla,
        load_blocks_to_device_buffer_mha,
        load_cache_to_device_buffer_dsv4_mla,
        load_cache_to_device_buffer_mla,
        load_cache_to_device_buffer_spec_mla,
        SPEC_SWAP_MAX_OCCURRENCES,
        SPEC_SWAP_MAX_STEPS,
    )

from sglang.kernels.ops.kvcache.hisparse_resident import (
    mask_resident_seq_lens,
    resident_page_table,
    translate_resident_rows,
)
from sglang.srt.configs.model_config import dsa_layer_skips_topk, is_deepseek_dsa
from sglang.srt.environ import envs
from sglang.srt.layers.attention.graph_variants import HISPARSE_RESIDENT
from sglang.srt.managers.hisparse_paging import HiSparsePagingManager
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
    ReqList,
    move,
)
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MiniMaxSparseKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.stream_handoff import handed_over, staged
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.mha import HiSparseMHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.model_executor.runner_utils.capture_mode import (
    get_capture_attention_variant,
)
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.kernels.ops.kvcache.hisparse import HiSparseSpecState
    from sglang.srt.managers.schedule_batch import ScheduleBatch

device_module = get_device_module()

_is_hip = is_hip()
_is_xpu = is_xpu()

logger = logging.getLogger(__name__)


class HiSparseAct(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    req: Req


class HiSparseTokenStats(NamedTuple):
    device_tokens: int
    device_token_usage: float
    host_tokens: int
    host_token_usage: float


def resolve_shared_index_layers(
    *,
    hf_text_config,
) -> Optional[List[bool]]:
    """Per-layer "reuses the previous layer's top-k index" pattern, or None.

    Mirrors DeepseekV2AttentionMLA's skip_topk derivation (index_topk_pattern /
    index_topk_freq / cli_factor); None when the model has no sharing or the
    prefetch cannot run (pipeline parallelism or the kill-switch).
    """
    if not is_deepseek_dsa(hf_text_config):
        return None
    num_layers = hf_text_config.num_hidden_layers
    cli_factor = getattr(hf_text_config, "cli_factor", 1) or 1
    if cli_factor > 1:
        pattern = [i % cli_factor != 0 for i in range(num_layers)]
    else:
        pattern = [dsa_layer_skips_topk(hf_text_config, i) for i in range(num_layers)]
    if not any(pattern):
        return None
    # Speculative decoding is supported by the HiSparse multi-step swap path;
    # only pipeline parallelism still prevents the shared-index prefetch plan.
    if get_parallel().pp_size != 1:
        logger.warning(
            "HiSparse shared-index prefetch is unsupported under pipeline "
            "parallelism; falling back to synchronous swap-in."
        )
        return None
    if envs.SGLANG_DISABLE_HISPARSE_PREFETCH.get():
        logger.info(
            "HiSparse shared-index prefetch disabled via "
            "SGLANG_DISABLE_HISPARSE_PREFETCH; using synchronous swap-in."
        )
        return None
    return pattern


def _build_prefetch_groups(
    is_shared_index_layer: List[bool],
) -> Tuple[Dict[int, List[int]], List[int]]:
    """Group consecutive shared-index (skip) layers under their anchor layer.

    Returns (groups, slot): anchor layer_id -> ordered skip layers, and each
    skip layer's position in its group (indexes the per-slot prefetch events).
    """
    groups: Dict[int, List[int]] = {}
    slot = [0] * len(is_shared_index_layer)
    anchor = None
    for i, is_shared in enumerate(is_shared_index_layer):
        if not is_shared:
            anchor = i  # compute layer; anchors the skip layers after it
            continue
        assert anchor is not None, (
            f"shared-index (skip) layer {i} has no preceding compute layer; "
            "the model's index-topk pattern is invalid"
        )
        group = groups.setdefault(anchor, [])
        slot[i] = len(group)
        group.append(i)
    return groups, slot


class HiSparseSpecSwapManager:
    """Own HiSparse speculative-swap state and request lifecycle."""

    HASH_MULTIPLIER = 2654435761
    MIN_SCRATCH_CAPACITY = 1024
    NUM_METADATA_VALUES_PER_OCCURRENCE = 5
    NUM_COUNTERS_PER_REQUEST = 4
    # Arbitrary: more blocks than arrivals between two finishes ever need.
    MAX_SPARE_SCRATCH = 16
    _spare_scratch = None  # (class default for managers built without __init__)

    def __init__(
        self,
        coordinator: HiSparseCoordinator,
        *,
        num_draft_tokens: int,
    ) -> None:
        self._coordinator = weakref.proxy(coordinator)
        self.enabled = num_draft_tokens > 0
        self.num_draft_tokens = num_draft_tokens
        self.scratch_capacity = 0
        self._scratch_reqs: set[int] = set()
        # Scratch blocks of ended resident requests, handed to the next arrivals:
        # an allocation below the band reads the free list back.
        self._spare_scratch: list = []

        layer_num = coordinator.mem_pool_device.layer_num
        max_num_req_slots = coordinator.req_to_token_pool.req_to_token.shape[0]
        self.req_to_scratch = torch.empty(
            (layer_num, max_num_req_slots, 0),
            dtype=torch.int32,
            device=coordinator.device,
        )
        self.states: tuple[HiSparseSpecState, ...] = ()
        self.top_k_device_locs: Optional[torch.Tensor] = None
        if not self.enabled:
            return

        if _is_xpu:
            raise ValueError("HiSparse speculative swap is unsupported on XPU.")
        if coordinator.is_dsv4_hisparse:
            raise ValueError("HiSparse spec swap currently supports DSA caches only.")
        if not 2 <= num_draft_tokens <= SPEC_SWAP_MAX_STEPS or coordinator.top_k < 1024:
            raise ValueError(
                f"HiSparse spec swap requires 2-{SPEC_SWAP_MAX_STEPS} draft tokens "
                "and top_k >= 1024."
            )
        workspace_capacity = num_draft_tokens * coordinator.top_k
        if workspace_capacity > SPEC_SWAP_MAX_OCCURRENCES:
            raise ValueError(
                "HiSparse spec swap supports at most "
                f"{SPEC_SWAP_MAX_OCCURRENCES} occurrences."
            )
        if max_num_req_slots > coordinator.device_buffer_size:
            raise ValueError(
                "HiSparse spec request capacity must not exceed device_buffer_size."
            )

        self.scratch_capacity = min(
            workspace_capacity,
            max(
                self.MIN_SCRATCH_CAPACITY,
                workspace_capacity - coordinator.device_buffer_size,
            ),
        )
        hash_size = 1 << max(1, (2 * coordinator.device_buffer_size - 1).bit_length())
        self._cache_index = torch.full(
            (layer_num, max_num_req_slots, 2, hash_size),
            -1,
            dtype=torch.int64,
            device=coordinator.device,
        )
        self._cache_policy = torch.zeros(
            (layer_num, max_num_req_slots + 1, coordinator.device_buffer_size),
            dtype=torch.int32,
            device=coordinator.device,
        )
        metadata_width = max(
            self.NUM_COUNTERS_PER_REQUEST * max_num_req_slots,
            self.NUM_METADATA_VALUES_PER_OCCURRENCE * workspace_capacity,
        )
        self._scratch_state = torch.full(
            (max_num_req_slots + 1, metadata_width),
            -1,
            dtype=torch.int32,
            device=coordinator.device,
        )
        self._scratch_state[0].zero_()
        self.req_to_scratch = torch.zeros(
            (layer_num, max_num_req_slots, self.scratch_capacity),
            dtype=torch.int32,
            device=coordinator.device,
        )
        self.states = tuple(
            HiSparseSpecState(
                cache_index=self._cache_index[layer_id],
                cache_policy=self._cache_policy[layer_id],
                scratch_locs=self.req_to_scratch[layer_id],
                scratch_state=self._scratch_state,
            )
            for layer_id in range(layer_num)
        )

        tokens = torch.arange(
            coordinator.device_buffer_size,
            dtype=torch.int64,
            device=coordinator.device,
        )
        self._hash_slots = ((tokens * self.HASH_MULTIPLIER) & (hash_size - 1)).to(
            torch.long
        )
        self._hash_entries = (tokens << 32) | tokens
        self.top_k_device_locs = torch.full(
            (max_num_req_slots, num_draft_tokens, coordinator.top_k),
            -1,
            dtype=torch.int32,
            device=coordinator.device,
        )

    @property
    def miss_plan_capacity(self) -> int:
        return self._coordinator.top_k * max(1, self.num_draft_tokens)

    def invalidate_cache(self, req_pool_idx: int) -> None:
        if self.enabled:
            self._cache_index[:, req_pool_idx].fill_(-1)

    def reset(self, req_pool_idx: int) -> None:
        if not self.enabled:
            return
        self._cache_index[:, req_pool_idx].fill_(-1)
        self._cache_index[:, req_pool_idx, 0, self._hash_slots] = (
            self._hash_entries.view(1, -1)
        )
        self._cache_policy[:, 0, req_pool_idx].zero_()
        self._cache_policy[:, req_pool_idx + 1].zero_()
        max_num_req_slots = self.req_to_scratch.shape[1]
        self._scratch_state[
            0,
            req_pool_idx : self.NUM_COUNTERS_PER_REQUEST
            * max_num_req_slots : max_num_req_slots,
        ] = 0
        self._scratch_state[req_pool_idx + 1].fill_(-1)

    def allocate_scratch(self, req_pool_idx: int) -> None:
        """Reserve one request's graph-stable swap workspace at admission."""
        if not self.enabled or req_pool_idx in self._scratch_reqs:
            return
        if self._spare_scratch:
            # Kept by a release on its own stream; this copy is queued behind the batch in flight.
            self.req_to_scratch[:, req_pool_idx].copy_(
                handed_over(self._spare_scratch.pop())
            )
            self._scratch_reqs.add(req_pool_idx)
            return
        allocator = self._coordinator.token_to_kv_pool_allocator
        if (  # admission counted resident requests' pages as free; move pages to host to free them
            getattr(self._coordinator, "hs_spec_res", False)
            is True  # a bool; SGLang's unit tests pass MagicMocks
            and allocator.hisparse_attn_allocator.available_size()
            < self.scratch_capacity
        ):
            self._coordinator.hs_spec_reclaims[0] += 1
            self._coordinator.reclaim_device(self.scratch_capacity)
        if isinstance(
            getattr(self._coordinator, "hs_el", None), dict
        ):  # scratch below the band (a dict; tests pass MagicMocks)
            scratch_locs = allocator._hs_el_alloc_below_or_evict(self.scratch_capacity)
        else:
            scratch_locs = allocator.hisparse_attn_allocator.alloc(
                self.scratch_capacity
            )
        if scratch_locs is None:
            raise RuntimeError(
                "HiSparse spec failed to allocate "
                f"{self.scratch_capacity} scratch slots."
            )
        # Physical slots are shared by all KV layers, but every full-index
        # layer rotates those IDs between persistent cache and scratch
        # independently. Keep per-layer location views so one layer cannot
        # overwrite the next layer's ownership state.
        self.req_to_scratch[:, req_pool_idx].copy_(scratch_locs.to(torch.int32))
        self._scratch_reqs.add(req_pool_idx)

    def free_unrotated_scratch(self, req_pool_idx: int) -> None:
        """Free scratch before any swap kernel has changed its ownership."""
        if not self.enabled or req_pool_idx not in self._scratch_reqs:
            return
        if (
            self._spare_scratch is not None
            and self._coordinator.enable_resident is True
            and len(self._spare_scratch) < self.MAX_SPARE_SCRATCH
        ):
            # Cloned on the stream of the release that ends the request; read by the next arrival's stream.
            self._spare_scratch.append(
                staged(self.req_to_scratch[0, req_pool_idx].clone())
            )
            self.clear_scratch(req_pool_idx)
            return
        self._coordinator.token_to_kv_pool_allocator.free_hisparse_indices(
            self.req_to_scratch[0, req_pool_idx].clone()
        )
        self.clear_scratch(req_pool_idx)

    def free_spare_scratch(self) -> bool:
        """Device memory is short: the kept scratch blocks go back to the allocator.
        Returns whether there were any."""
        if not self._spare_scratch:
            return False
        allocator = self._coordinator.token_to_kv_pool_allocator
        P = self._coordinator.page_size
        while self._spare_scratch:
            # A block that was never rotated: whole pages, in page order. Kept by a release on its own
            # stream; nothing here reads back, so the page ids are computed when this stream gets to them.
            block = handed_over(self._spare_scratch.pop())
            allocator.free_device_pages(block[::P] // P)
        return True

    def spare_scratch_slots(self) -> int:
        """Device slots of the kept blocks: free for admission, since
        `free_spare_scratch` returns them when slots are short."""
        return len(self._spare_scratch or ()) * self.scratch_capacity

    def forget_spare_scratch(self) -> None:
        """The allocator rebuilt its free list from every page: the kept blocks are in it."""
        if self._spare_scratch:
            self._spare_scratch.clear()

    def clear_scratch(self, req_pool_idx: int) -> None:
        if not self.enabled or req_pool_idx not in self._scratch_reqs:
            return
        self.req_to_scratch[:, req_pool_idx].zero_()
        self._scratch_reqs.remove(req_pool_idx)

    def extend_owned_locs(
        self, req_pool_idx: int, owned_locs: torch.Tensor
    ) -> torch.Tensor:
        if not self.enabled or req_pool_idx not in self._scratch_reqs:
            return owned_locs
        return torch.cat([owned_locs, self.req_to_scratch[:, req_pool_idx].reshape(-1)])

    def prepare_verify(self, batch: ScheduleBatch) -> None:
        """Bind target-verify KV writes to the side buffer's extra page."""
        coordinator = self._coordinator
        req_pool_indices = batch.req_pool_indices
        verify_cache_locs = batch.out_cache_loc
        start_positions = batch.seq_lens

        if not self.enabled:
            raise RuntimeError("HiSparse spec is not initialized.")
        if self.num_draft_tokens > coordinator.page_size - 1:
            raise ValueError(
                f"HiSparse spec needs {self.num_draft_tokens} verify slots, but the "
                f"extra page has only {coordinator.page_size - 1} usable slots."
            )
        expected_slots = req_pool_indices.numel() * self.num_draft_tokens
        if verify_cache_locs.numel() != expected_slots:
            raise ValueError(
                f"HiSparse verify slot mismatch: expected {expected_slots}, "
                f"got {verify_cache_locs.numel()}."
            )
        reqs = batch.reqs
        coordinator._hs_verify_sel = None
        if (
            coordinator.hs_spec_res
        ):  # resident rows write their verify KV to their own mirrored slots
            res_rows = coordinator._hs_spec_res_prepare(batch)
            if len(res_rows) == len(reqs):
                coordinator._hs_verify_sel = len(
                    reqs
                )  # an all-resident round: the resident graph
                return
            if res_rows:
                rs = set(res_rows)
                buf_rows = [i for i in range(len(reqs)) if i not in rs]
                idx = coordinator._gpu_rows(buf_rows)
                reqs = [reqs[i] for i in buf_rows]
                req_pool_indices = req_pool_indices[idx]
                start_positions = start_positions[idx]
                verify_cache_locs = verify_cache_locs.reshape(
                    -1, self.num_draft_tokens
                )[idx].reshape(-1)
                expected_slots = len(buf_rows) * self.num_draft_tokens

        if coordinator.hs_exclusive is True:
            page = coordinator.page_size
            coordinator.paging.ensure_host(
                sum(
                    max(
                        -(-req.kv.kv_allocated_len // page) * page
                        - int(
                            coordinator.req_to_host_pool_allocated_len[
                                req.kv.req_pool_idx
                            ]
                        ),
                        0,
                    )
                    for req in reqs
                ),
                must=True,
            )
        # CPU-side logical reservations cover verify even when overlap lags seq_lens.
        for req in reqs:
            req_pool_idx = req.kv.req_pool_idx
            allocated_len = int(
                coordinator.req_to_host_pool_allocated_len[req_pool_idx]
            )
            if req.kv.kv_allocated_len > allocated_len:
                coordinator.mem_pool_host.alloc_paged_token_slots(
                    coordinator.req_to_host_pool,
                    coordinator.req_to_host_pool_allocated_len,
                    req_pool_idx,
                    allocated_len,
                    req.kv.kv_allocated_len - allocated_len,
                )

        extra_start = coordinator.device_buffer_size + 1
        row_indices = torch.repeat_interleave(req_pool_indices, self.num_draft_tokens)
        offsets = (
            torch.arange(
                expected_slots, dtype=torch.int64, device=req_pool_indices.device
            )
            % self.num_draft_tokens
        )
        token_positions = (
            torch.repeat_interleave(
                start_positions.to(torch.int64), self.num_draft_tokens
            )
            + offsets
        )
        columns = extra_start + offsets
        device_locs = coordinator.req_to_device_buffer[row_indices, columns]
        coordinator.req_device_buffer_tokens[:, row_indices, columns] = (
            token_positions.to(torch.int32).unsqueeze(0)
        )
        coordinator.req_device_buffer_token_locs[:, row_indices, columns] = (
            device_locs.to(torch.int32).unsqueeze(0)
        )
        coordinator.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping[
            verify_cache_locs
        ] = device_locs

    def _backup_device_locs_to_host(
        self, host_locs: torch.Tensor, device_locs: torch.Tensor
    ) -> None:
        if host_locs.numel() == 0:
            return
        coordinator = self._coordinator
        coordinator.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        device_locs = device_locs.contiguous()
        with device_module.stream(coordinator.decode_backup_stream):
            coordinator.decode_backup_stream.wait_stream(schedule_stream)
            if coordinator.decode_producer_stream is not None:
                coordinator.decode_backup_stream.wait_stream(
                    coordinator.decode_producer_stream
                )
            coordinator.mem_pool_host.backup_from_device_all_layer(
                coordinator.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            if host_locs.is_cuda:
                host_locs.record_stream(coordinator.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(coordinator.decode_backup_stream)
            coordinator._backup_done_event.record()
        coordinator._has_pending_backup = True
        coordinator.wait_for_pending_backup()

    def commit_accept_tokens(
        self, batch: ScheduleBatch, accept_indices: torch.Tensor
    ) -> None:
        """Persist accepted target-verify KV in host and the persistent buffer."""
        if batch.forward_mode.is_idle():
            return
        coordinator = self._coordinator
        req_pool_indices = batch.req_pool_indices[:, None]
        verify_cache_locs = batch.out_cache_loc

        if verify_cache_locs.numel() == 0:
            return
        batch_size = req_pool_indices.numel()
        if batch_size == 0 or verify_cache_locs.numel() % batch_size != 0:
            raise ValueError("HiSparse verify cache locations are not request-aligned.")

        reqs = batch.reqs
        seq_lens = batch.seq_lens
        row_ids = None  # batch rows taking the stock commit (None = all)
        all_verify_locs = verify_cache_locs
        if coordinator.hs_spec_res:
            res = coordinator._resident
            res_rows = [i for i, r in enumerate(reqs) if r.kv.req_pool_idx in res]
            if len(res_rows) == len(reqs):
                # A resident request's accepted KV is already in its own slots;
                # it is copied to host when its page leaves the device pool.
                return
            if res_rows:
                rs = set(res_rows)
                buf_rows = [i for i in range(len(reqs)) if i not in rs]
                row_ids = coordinator._gpu_rows(buf_rows)
                reqs = [reqs[i] for i in buf_rows]
                req_pool_indices = req_pool_indices[row_ids]
                seq_lens = seq_lens[row_ids]
                accept_indices = accept_indices[row_ids]
                verify_cache_locs = all_verify_locs.reshape(batch_size, -1)[
                    row_ids
                ].reshape(-1)

        valid = accept_indices >= 0
        counts = valid.sum(dim=1)
        steps = torch.arange(accept_indices.shape[1], device=accept_indices.device)
        positions = seq_lens[:, None].to(torch.int64) + steps
        mapping = (
            coordinator.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
        )
        # Keep a fixed width: boolean indexing and GPU-sized repeat_interleave sync.
        verify_width = all_verify_locs.numel() // batch_size
        rows_full = (  # accept indices index the whole batch's verify tokens
            torch.arange(batch_size, device=accept_indices.device)
            if row_ids is None
            else row_ids
        )
        row_offsets = rows_full[:, None] * verify_width
        source_offsets = torch.where(valid, accept_indices, row_offsets + steps)
        source_device_locs = mapping[all_verify_locs[source_offsets]].clone()
        canonical_locs = coordinator.req_to_token_pool.req_to_token[
            req_pool_indices, positions
        ]

        # Keep hot tokens and each request's newest token in persistent slots.
        cache_mask = valid & (
            (positions < coordinator.device_buffer_size)
            | (steps == counts[:, None] - 1)
        )
        cache_positions = torch.where(
            cache_mask,
            positions.clamp(max=coordinator.device_buffer_size),
            coordinator.device_buffer_size + 1 + steps,
        )
        cache_locs = coordinator.req_to_device_buffer[req_pool_indices, cache_positions]
        destination_locs = torch.where(cache_mask, cache_locs, source_device_locs)
        coordinator.mem_pool_device.transfer_values_on_device(
            dst_indices=destination_locs.flatten(),
            src_indices=source_device_locs.flatten(),
        )

        # Rejected padding goes to this request's reserved, not-yet-visible host slots.
        host_locs = coordinator.req_to_host_pool[req_pool_indices, positions]
        self._backup_device_locs_to_host(
            host_locs.flatten(), destination_locs.flatten()
        )

        # Scalar indexed assignment performs a blocking CPU-to-CUDA copy.
        mapping.index_fill_(0, verify_cache_locs, 0)
        mapping[canonical_locs] = torch.where(cache_mask, cache_locs, 0)
        coordinator.req_device_buffer_tokens[:, req_pool_indices, cache_positions] = (
            torch.where(
                cache_mask,
                positions.to(torch.int32),
                coordinator.req_device_buffer_tokens[
                    :, req_pool_indices, cache_positions
                ],
            )
        )
        coordinator.req_device_buffer_token_locs[
            :, req_pool_indices, cache_positions
        ] = torch.where(
            cache_mask,
            cache_locs.to(torch.int32),
            coordinator.req_device_buffer_token_locs[
                :, req_pool_indices, cache_positions
            ],
        )
        for req in reqs:
            coordinator._skip_first_backup[req.kv.req_pool_idx] = True

    def swap_in(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        **plan,
    ) -> torch.Tensor:
        if not self.enabled:
            raise RuntimeError("HiSparse spec swap is not initialized.")
        num_reqs = req_pool_indices.size(0)
        num_steps, num_top_k = top_k_result.shape[1:]
        if num_steps != self.num_draft_tokens:
            raise ValueError(
                f"HiSparse spec step mismatch: expected {self.num_draft_tokens}, "
                f"got {num_steps}."
            )
        assert self.top_k_device_locs is not None
        top_k_indices = self.top_k_device_locs[:num_reqs, :num_steps, :num_top_k]
        coordinator = self._coordinator
        load_cache_to_device_buffer_spec_mla(
            top_k_tokens=top_k_result,
            device_buffer_tokens=coordinator.req_device_buffer_tokens[layer_id],
            host_cache_locs=coordinator.req_to_host_pool,
            device_buffer_locs=coordinator.req_device_buffer_token_locs[layer_id],
            host_cache=coordinator.mem_pool_host.kv_buffer[layer_id],
            device_buffer=coordinator.mem_pool_device.kv_buffer[layer_id],
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            state=self.states[layer_id],
            num_real_reqs=coordinator.num_real_reqs,
            **plan,
        )
        return top_k_indices

    def output_locs(self, top_k_result: torch.Tensor, num_reqs: int) -> torch.Tensor:
        if not self.enabled or self.top_k_device_locs is None:
            raise RuntimeError("HiSparse spec swap is not initialized.")
        return self.top_k_device_locs[
            :num_reqs, : top_k_result.shape[1], : top_k_result.shape[2]
        ]


class _HsUnifiedMLAHost(MLATokenToKVPoolHost):
    """the HiSparse host pool over the host part of the device pool's unified VMM range (no allocation, no
    registration: the range is already pinned and mapped for the GPU, so kernels use its addresses directly)."""

    def __init__(self, *args, hs_device_pool=None, **kwargs):
        self._hs_device_pool = hs_device_pool
        super().__init__(*args, **kwargs)

    def _allocate_host_buffer(self) -> None:
        assert self.layout == "layer_first", self.layout
        # what init_kv_buffer sets besides allocating
        self.token_stride_size = self.kv_cache_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num
        self.kv_buffer = self._hs_device_pool.hs_host_region(
            self.size, self.dtype, self.kv_cache_dim
        )
        self.fd = None
        logger.info(
            "HiSparse unified VA: host pool of %d tokens mapped in the device pool's range",
            self.size,
        )

    def destroy(self):
        return


class HiSparseCoordinator:
    hs_spec_res = False  # (class default for coordinators built without __init__)
    _recv_buffer_hold = None  # (class default) req slot -> buffer slots kept for a PD transfer
    hs_el = None  # (class default)
    _hs_verify_sel = None  # (class default)
    enable_resident = False
    hs_exclusive = False
    hs_host_scarce = False
    paging = None
    _resident = types.MappingProxyType({})
    _resident_steps = 0
    _release_queue = None  # (class default)

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: Union[
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        ],
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group,
        host_to_device_ratio: int = 2,
        swap_in_block_size: int = 960,
        shared_index_layers: Optional[List[bool]] = None,
        num_draft_tokens: int = 0,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.device = device
        self.swap_in_block_size = swap_in_block_size
        # Timing probe: skip the host->device KV bytes to measure the "IO is
        # free" floor. Produces garbage output; benchmarking only.
        self.skip_io = envs.SGLANG_DEBUG_HISPARSE_SKIP_IO.get()
        if _is_xpu and self.skip_io:
            raise ValueError(
                "SGLANG_DEBUG_HISPARSE_SKIP_IO is unsupported on XPU: the AOT swap-in "
                "ops do not accept skip_io, so timings would silently include the KV "
                "copy."
            )
        self.compress_ratio = self.token_to_kv_pool_allocator.compress_ratio

        kvcache = self.token_to_kv_pool_allocator.get_kvcache()
        self.is_dsv4_hisparse = isinstance(
            self.token_to_kv_pool_allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator
        )
        self.is_m3_hisparse = isinstance(kvcache, MiniMaxSparseKVPool)
        self.mem_pool_device = self.token_to_kv_pool_allocator.get_hisparse_kvcache()
        # Paged residency: the host pool lives in the device pool's address range.
        self._hs_unified = (
            isinstance(self.mem_pool_device, HiSparseDSATokenToKVPool)
            and self.mem_pool_device._hs_va is not None
        )
        if self.is_dsv4_hisparse:
            page_size = self.mem_pool_device.page_size
            num_host_pages = (
                self.token_to_kv_pool_allocator.size_full // self.compress_ratio
                + page_size
                - 1
            ) // page_size
            self.mem_pool_host = DeepSeekV4PagedHostPool(
                pool_name="dsv4_hisparse_c4",
                device_buffers=self.mem_pool_device.kv_buffer,
                item_bytes=self.mem_pool_device.bytes_per_page_padded,
                num_host_pages=num_host_pages,
                slot_page_size=page_size,
                layout="layer_first",
            )
            self.item_size_bytes = (
                self.mem_pool_device.kv_cache_total_dim
                * self.mem_pool_device.store_dtype.itemsize
            )
        else:
            assert isinstance(
                self.token_to_kv_pool_allocator, HiSparseTokenToKVPoolAllocator
            )
            if self.is_m3_hisparse:
                assert self.mem_pool_device.head_num == 1, (
                    "MiniMax M3 HiSparse requires one KV head per TP rank, "
                    f"got {self.mem_pool_device.head_num}. Increase the "
                    "tensor-parallel size."
                )
                self.mem_pool_host = HiSparseMHATokenToKVPoolHost(
                    buffer_info=self.mem_pool_device.get_device_pool_infos()[
                        0
                    ].buffer_info,
                    device_capacity_tokens=self.mem_pool_device.size
                    if self.mem_pool_device.host_capacity_tokens is None
                    else self.mem_pool_device.host_capacity_tokens,
                    start_layer=self.mem_pool_device.start_layer,
                    end_layer=self.mem_pool_device.end_layer,
                    host_to_device_ratio=host_to_device_ratio,
                    page_size=self.mem_pool_device.page_size,
                )
                self.item_size_bytes = self.mem_pool_device.bytes_per_token_k
            else:
                # In the shared address range the host pool takes its buffer
                # from the device pool's mapping.
                host_cls = (
                    _HsUnifiedMLAHost if self._hs_unified else MLATokenToKVPoolHost
                )
                host_kw = (
                    dict(pin_memory=False, hs_device_pool=self.mem_pool_device)
                    if self._hs_unified
                    else {}
                )
                if self._hs_unified:
                    # Exclusive placement may name a host pool below the logical pool.
                    host_to_device_ratio = self.mem_pool_device._hs_host_ratio
                self.mem_pool_host = host_cls(
                    buffer_info=self.mem_pool_device.get_device_pool_infos()[
                        0
                    ].buffer_info,
                    device_capacity_tokens=self.mem_pool_device.size
                    if self.mem_pool_device.host_capacity_tokens is None
                    else self.mem_pool_device.host_capacity_tokens,
                    start_layer=self.mem_pool_device.start_layer,
                    end_layer=self.mem_pool_device.end_layer,
                    target_layer_num=self.mem_pool_device.layer_num,
                    host_to_device_ratio=host_to_device_ratio,
                    host_size=0,
                    page_size=self.mem_pool_device.page_size,
                    layout="layer_first",
                    override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
                    **host_kw,
                )
                self.item_size_bytes = self.mem_pool_host.token_stride_size
        self.page_size = self.mem_pool_device.page_size
        # Model layer id -> layer index of the dense full pool; None when the
        # attention callers already pass pool layer indices.
        self._pool_layer_index: Optional[Dict[int, int]] = (
            kvcache.full_attention_layer_id_mapping
            if isinstance(kvcache, HybridLinearKVPool)
            else None
        )

        max_num_req_slots = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len
        max_compressed_context_len = (
            max_context_len + self.compress_ratio - 1
        ) // self.compress_ratio

        # to have an extra page for new tokens
        self.padded_buffer_size = (
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        self.req_to_device_buffer = torch.zeros(
            (max_num_req_slots, self.padded_buffer_size),
            dtype=torch.int64,
            device=device,
        )
        self.req_device_buffer_size = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )
        self.req_to_host_pool = torch.full(
            (max_num_req_slots, max_compressed_context_len + self.page_size),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.req_to_host_pool_allocated_len = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )

        self.write_staging_stream = device_module.Stream()
        self.decode_backup_stream = device_module.Stream()
        self.ack_staging_queue = IndexedReqList(
            ListKind.HISPARSE_STAGING, req_of=lambda x: x.req
        )
        self.decode_producer_stream = None
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # initialize data structures for swap-in kernel
        layer_num = self.mem_pool_device.layer_num
        self.req_device_buffer_tokens = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_device_buffer_token_locs = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._lru_init = torch.arange(
            self.device_buffer_size, dtype=torch.int16, device=device
        )
        self.lru_slots = (
            self._lru_init.view(1, 1, -1)
            .repeat(layer_num, max_num_req_slots, 1)
            .contiguous()
        )
        self.spec_swap = HiSparseSpecSwapManager(
            self,
            num_draft_tokens=num_draft_tokens,
        )
        if isinstance(self.token_to_kv_pool_allocator, HiSparseTokenToKVPoolAllocator):
            self.token_to_kv_pool_allocator.configure_spec_scratch(
                self.spec_swap.scratch_capacity,
                device_buffer_size=self.padded_buffer_size,
            )
        self._device_buffer_arange_i32 = torch.arange(
            self.device_buffer_size, dtype=torch.int32, device=device
        )

        # Pre-allocated output buffer for swap-in (CUDA-graph safe). MiniMax
        # selects blocks, so its flattened token-slot output can occupy any
        # prefix up to the full device working-set size.
        swap_output_width = (
            self.device_buffer_size if self.is_m3_hisparse else self.top_k
        )
        self.top_k_device_locs_buffer = torch.full(
            (max_num_req_slots, swap_output_width),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.raw_indices_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        # Scalar tensor: number of real (non-padded) requests in the batch.
        # Updated before each graph replay so padded blocks early-return.
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

        # Paged residency: a request longer than the hot buffer keeps its KV where
        # the prefill wrote it; its rows are read there instead of being swapped in.
        self.enable_resident = envs.SGLANG_HISPARSE_PAGED.get()
        if self.enable_resident and not self._hs_unified:
            raise ValueError(
                "SGLANG_HISPARSE_PAGED needs the DSA HiSparse pool on its mapped "
                "address range (not DeepSeek V4, MiniMax or a hybrid-linear pool)."
            )
        # Exclusive placement: a page in the device pool holds a host slot only
        # until the slot is needed, so the logical pool may exceed the host pool.
        self.hs_exclusive = bool(envs.SGLANG_HISPARSE_PAGED_EXCLUSIVE.get())
        if self.hs_exclusive and not self.enable_resident:
            raise ValueError(
                "SGLANG_HISPARSE_PAGED_EXCLUSIVE needs SGLANG_HISPARSE_PAGED."
            )
        # Under speculative decoding a resident request's verify tokens are
        # written to its own mirrored device pages.
        self.hs_spec_res = self.enable_resident and num_draft_tokens > 0
        self._hs_dev_len = {}  # req_pool_idx -> positions with device slots (mirrored logical pages)
        self._hs_spec_steps = [0, 0, 0]  # verify steps; resident rows; buffered rows
        self.hs_spec_reclaims = [
            0,
            0,
        ]  # reclaims before a scratch / a buffer allocation
        # The pool's mapped ranges and their limits; None for a pool without them.
        allocator = self.token_to_kv_pool_allocator
        self.hs_el = (
            allocator._hs_el
            if isinstance(allocator, HiSparseTokenToKVPoolAllocator)
            else None
        )
        if self.enable_resident and self.hs_el is None:
            # A wrapper pool (hybrid-linear) hides the inner pool's ranges from the
            # allocator, which would then never grow them.
            raise ValueError(
                "SGLANG_HISPARSE_PAGED needs the allocator to hold the DSA HiSparse "
                "pool itself, not a wrapper around it."
            )
        self.hs_el_stats = [
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # band grows, band shrinks, pages relocated, buffer pages moved, evictions for room, rounds
        if self.hs_el is not None:
            self.hs_el_grow_band(log=True)
            # Every physical chunk the band and the index keys can ever map is created
            # now and pooled: a cuMemCreate at run time competes with PyTorch's caching
            # allocator for the last of device memory and can fail.
            el = self.hs_el
            kv, ik = el["kv"], el["ik"]
            # the most ever mapped at once: band + keys <= E, except that keys mapped for the whole logical pool round
            # up to whole 2 MiB chunks per layer (up to one chunk per layer above E, with the band at zero)
            total = max(-(-el["E"] // kv.g), ik.stride // ik.g * ik.layers)
            have = kv.handles() + ik.mapped // ik.g * ik.layers + kv.x.pooled(kv.g)
            if total > have:
                kv.x.reserve_pool(kv.dev, kv.g, total - have)
            el["creates0"] = int(kv.x.creates())
            logger.info(
                "HiSparse elastic: %d physical chunks of %d MiB for the %.1f GiB reservation created up front "
                "(%d pooled); cuMemCreate calls %d",
                total,
                kv.g >> 20,
                el["E"] / 2**30,
                kv.x.pooled(kv.g),
                el["creates0"],
            )
        self._hs_arange_page = torch.arange(
            self.page_size, dtype=torch.int64, device=device
        )
        # Load every variant of the device-to-device copy kernel while device memory
        # has room (num_locs_upper is a constexpr: one variant per power of two up to
        # 256); a variant first needed with the pool full fails to load. Identity
        # copies of the padding slot 0, which no request owns.
        if self.enable_resident:
            for n in (1, 2, 3, 5, 9, 16, 17, 32, 33, 48, 65, 80, 129, 144):
                z = torch.zeros(n, dtype=torch.int64, device=device)
                self.mem_pool_device.transfer_values_on_device(z, z)
        self.req_resident = torch.zeros(
            max_num_req_slots, dtype=torch.int8, device=device
        )
        self._resident: dict = {}  # req_pool_idx -> Req
        self._resident_steps = 0
        # Ended requests whose pages the launched forward may still touch:
        # (req, events that end that use, the caller's release), oldest first.
        self._release_queue: list = []
        self._release_pending: set = set()  # their req_pool_idx
        self._frame_wait_since: dict = {}  # rid -> first refusal (perf_counter)
        self._frame_waits: list = []  # ms
        self._frame_wait_count = 0
        # Their release runs here: the scheduler's stream waits for each launched
        # forward (its write-after-read barrier), and so would every host read on it.
        self._release_stream = device_module.Stream()
        # Batches whose requests are all resident replay their own decode graph.
        self._hs_verify_sel = (
            None  # rows of the last verify preparation when every row was resident
        )
        self._hs_pt_batch = None
        self._hs_pt_seq = None
        self._hs_sel = [0, 0]
        self._hs_pt_memo = None
        self._hs_staged_len = {}  # req_pool_idx -> tokens backed up by staging (the host copy's prefix)
        self._hs_lean = [
            0,
            0,
        ]  # decode steps prepared for all-resident batches, for mixed batches
        if self.hs_spec_res:
            logger.info(
                "HiSparse: resident requests under speculative decoding (mirrored device pages, lazy backup)"
            )
        if self.enable_resident:
            logger.info(
                "HiSparse: paged residency on (requests longer than the %d-token hot "
                "buffer keep their KV in the device pool and on host pages)",
                self.device_buffer_size,
            )

        # CPU flag: True means "skip backup on the next decode step" because
        # staging already backed up all prefill tokens.  Cleared after one step.
        self._skip_first_backup = [False] * max_num_req_slots

        self._init_shared_index_prefetch(
            shared_index_layers=shared_index_layers,
            layer_num=layer_num,
            max_num_req_slots=max_num_req_slots,
        )
        if self.enable_resident:
            # Built last: it reads the shared-index groups and the band's limits.
            self.paging = HiSparsePagingManager(self)
            if self.hs_exclusive:
                # Not every token has a host slot: a prompt that stays buffered
                # is charged its host rows at admission and takes them then.
                self.hs_host_scarce = (
                    self.mem_pool_host.size < self.token_to_kv_pool_allocator.size_full
                )
                if self.hs_host_scarce:
                    self.token_to_kv_pool_allocator._buffered_prompt_tokens = (
                        self.device_buffer_size
                    )
                logger.info(
                    "HiSparse paged: exclusive placement on (logical pool %d tokens, "
                    "host pool %d tokens, device pool %d tokens + band)",
                    self.token_to_kv_pool_allocator.size_full,
                    self.mem_pool_host.size,
                    self.mem_pool_device.size,
                )
            # From here on admission counts pages that can move to host as free.
            self.token_to_kv_pool_allocator.set_reclaimer(self)

    def _init_shared_index_prefetch(
        self,
        shared_index_layers: Optional[List[bool]],
        layer_num: int,
        max_num_req_slots: int,
    ) -> None:
        """Set up the plan-then-IO prefetch for shared-index (IndexShare) models:
        the anchor's kernel records its miss plan and skip layers replay it on
        `prefetch_stream`, overlapping their IO with the intervening compute."""
        if shared_index_layers is not None and len(shared_index_layers) != layer_num:
            # Attention-layer count differs from num_hidden_layers (e.g. Longcat
            # doubles it): pattern would be misindexed, fall back to synchronous.
            logger.warning(
                "HiSparse shared-index prefetch disabled: pattern length %d != "
                "KV pool layer_num %d; using synchronous swap-in.",
                len(shared_index_layers),
                layer_num,
            )
            shared_index_layers = None
        if shared_index_layers is not None and _is_xpu:
            logger.warning(
                "HiSparse shared-index prefetch disabled on XPU: "
                "copy_cache_planned_mla is unavailable; using synchronous swap-in."
            )
            shared_index_layers = None
        self._is_shared_index_layer = list(shared_index_layers or [False] * layer_num)
        self.enable_prefetch = any(self._is_shared_index_layer)
        self._prefetch_groups, self._prefetch_slot = _build_prefetch_groups(
            self._is_shared_index_layer
        )
        if not self.enable_prefetch:
            return

        # Small fixed grid for the copy-only kernel: low SM footprint so the
        # copies overlap compute with little contention.
        self._prefetch_copy_blocks = 4
        max_group_size = max(len(g) for g in self._prefetch_groups.values())
        self.prefetch_stream = device_module.Stream()
        self._prefetch_events = [device_module.Event() for _ in range(max_group_size)]
        # Plan recorded by the current anchor, replayed by its skip layers. One
        # buffer set suffices: the last skip layer's event wait orders the next
        # anchor's writes after this group's copies.
        miss_plan_capacity = self.spec_swap.miss_plan_capacity
        self._miss_src = torch.zeros(
            (max_num_req_slots, miss_plan_capacity),
            dtype=torch.int64,
            device=self.device,
        )
        self._miss_dst = torch.zeros(
            (max_num_req_slots, miss_plan_capacity),
            dtype=torch.int32,
            device=self.device,
        )
        self._miss_count = torch.zeros(
            (max_num_req_slots,), dtype=torch.int32, device=self.device
        )
        logger.info(
            "HiSparse: shared-index prefetch (plan-then-IO) enabled; %d anchor "
            "group(s), %d skip layer(s) of %d total.",
            len(self._prefetch_groups),
            sum(self._is_shared_index_layer),
            layer_num,
        )

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    def allocate_spec_scratch(self, req_pool_indices: List[int]) -> None:
        for req_pool_idx in req_pool_indices:
            self.spec_swap.allocate_scratch(req_pool_idx)

    def hold_buffer_slots(self, req: Req, prefill_tokens: int) -> bool:
        """PD receive: keep, from pre-allocation on, the device slots the hot
        buffer of a prompt that stays buffered adds to its prefill pages.
        Admission charged them (`request_slot_reserve`). A local prefill takes
        them one step after that check; a transfer lands many pre-allocations
        later, and frames of transfers in flight cannot be evicted for it.
        False when they do not fit."""
        slots = self.token_to_kv_pool_allocator.alloc_buffer_extra(prefill_tokens)
        if slots is None:
            return False
        if slots.numel() > 0:
            if self._recv_buffer_hold is None:
                self._recv_buffer_hold = {}
            # Taken on the pre-allocating stream; given back by whatever stream ends or admits the request.
            self._recv_buffer_hold[req.kv.req_pool_idx] = staged(slots)
        return True

    def _give_back_buffer_hold(self, req_pool_idx: int) -> None:
        """The request builds its buffer now, or ends without one."""
        held = self._recv_buffer_hold
        if held:
            slots = held.pop(req_pool_idx, None)
            if slots is not None:
                self.token_to_kv_pool_allocator.free_buffer_extra(slots)

    def cancel_prealloc(self, req: Req) -> None:
        """PD receive: a pre-allocation that got no device frames gives back
        what it took with its request slot: speculative scratch, a buffered
        prompt's host rows, held buffer slots."""
        rid = req.kv.req_pool_idx
        self._give_back_buffer_hold(rid)
        self.spec_swap.free_unrotated_scratch(rid)
        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool, rid, self.req_to_host_pool_allocated_len[rid]
        )
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        self.req_to_host_pool[rid, :] = -1
        self.req_to_host_pool_allocated_len[rid] = 0

    def wait_frames_quiescent(self) -> None:
        """Block until no launched forward and no queued copy can touch a device
        frame handed out so far: for a writer that follows no local stream (a
        PD prefill writing a published destination)."""
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()
        device_module.current_stream().synchronize()

    def destroy(self) -> None:
        # Drain in-flight transfers so the buffer is idle, then unregister it.
        # See HostKVCache.destroy for why the explicit unregister matters.
        self.write_staging_stream.synchronize()
        self.decode_backup_stream.synchronize()
        if self.enable_prefetch:
            # Skip-layer copies read the pinned host pool on the prefetch stream.
            self.prefetch_stream.synchronize()
        self.mem_pool_host.destroy()

    def get_token_stats(self) -> HiSparseTokenStats:
        device_allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        device_capacity = device_allocator.size
        device_tokens = device_capacity - device_allocator.available_size()
        host_capacity = self.mem_pool_host.size
        host_tokens = host_capacity - self.mem_pool_host.available_size()
        return HiSparseTokenStats(
            device_tokens=device_tokens,
            device_token_usage=(
                device_tokens / device_capacity if device_capacity > 0 else 0.0
            ),
            host_tokens=host_tokens,
            host_token_usage=(
                host_tokens / host_capacity if host_capacity > 0 else 0.0
            ),
        )

    def admit_request_into_staging(self, req: Req, received: bool = False) -> None:
        """Start the host copy of a prefilled request and move it into staging.
        received: the prompt arrived by a PD transfer."""
        req.hisparse_staging = True

        full_kv_indices = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : req.extend_end
        ].to(dtype=torch.int64, copy=True)
        device_indices = (
            self.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                full_kv_indices
            )
        )

        prefill_len = len(device_indices)
        if (
            self.enable_resident
        ):  # the host copy's prefix if the request becomes resident
            self._hs_staged_len[req.kv.req_pool_idx] = prefill_len
        if self.hs_exclusive:
            prefill_len = self._exclusive_staged_len(req, prefill_len, received)
            device_indices = device_indices[:prefill_len]
        host_indices = self.mem_pool_host.alloc_paged_token_slots(
            self.req_to_host_pool,
            self.req_to_host_pool_allocated_len,
            req.kv.req_pool_idx,
            0,
            prefill_len,
        )

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                io_backend="kernel",
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        move([HiSparseAct(start_event, finish_event, req)], self.ack_staging_queue)

    def _exclusive_staged_len(self, req: Req, prefill_len: int, received: bool) -> int:
        """Exclusive placement: how much of a prefill is copied to host. All of a
        prompt that stays buffered; the whole pages of a resident one while that
        many host slots are free, else nothing (its pages then hold no host slot
        until they leave the device pool). Nothing of a resident prompt that
        arrived by a PD transfer: on a decode that copy runs against the
        transfers landing next to it."""
        if not self._can_stay_resident(req):
            page_end = -(-prefill_len // self.page_size) * self.page_size
            held = int(self.req_to_host_pool_allocated_len[req.kv.req_pool_idx])
            self.paging.ensure_host(page_end - held, must=True)
            return prefill_len
        whole = prefill_len // self.page_size * self.page_size
        # Free slots only: taking slots back from pages in the device pool would
        # copy every arrival to host and wait for the launched forward to do it.
        if not received and self.mem_pool_host.available_size() >= whole:
            return whole
        self.paging.xstats[5] += 1
        return 0

    def admit_request_direct(self, req: Req) -> None:
        """Direct-to-host path: KV data already resides in host pool via RDMA.

        Skips staging DMA entirely. Only allocates a small device buffer
        (4KB) for decode-time swap-in, then marks the request as ready.
        Host indices were already written to req_to_host_pool.

        Metadata fixups after alloc_device_buffer():
        - alloc_device_buffer() sets device_buffer_tokens = [0, 1, ..., buf_size-1],
          which tells the swap-in kernel that those tokens are cached in the device
          buffer.  In the staging path this is correct (prefill filled the buffer),
          but here the buffer is empty.
        """
        self.alloc_device_buffer(req)

        host_len = self.host_token_len(req.kv.kv_allocated_len)
        if host_len <= self.device_buffer_size:
            # Short sequences (seq_len <= device_buffer_size): the kernel fast path
            # returns device_buffer_locs directly without any host loading, so we
            # must preload all tokens from host pool into the device buffer
            # TODO(hzh0425): Optimize this.
            self._preload_to_device_buffer(req)
        else:
            # Long sequence: reset device_buffer_tokens to -1 so the kernel
            # sees all slots as empty -> every top-k lookup is a miss -> host load.
            self.req_device_buffer_tokens[
                :, req.kv.req_pool_idx, : self.device_buffer_size
            ] = -1
            self.spec_swap.invalidate_cache(req.kv.req_pool_idx)

        req.hisparse_staging = False
        self._skip_first_backup[req.kv.req_pool_idx] = True
        logger.debug("HiSparse: admitting request %s directly", req.rid)

    def host_token_len(self, kv_allocated_len: int) -> int:
        if self.is_dsv4_hisparse:
            return kv_allocated_len // self.compress_ratio
        return kv_allocated_len

    def _preload_to_device_buffer(self, req: Req) -> None:
        """Preload all tokens from host pool into the device buffer."""
        n = self.host_token_len(req.kv.kv_allocated_len)
        host_indices = self.req_to_host_pool[req.kv.req_pool_idx, :n]
        device_locs = self.req_to_device_buffer[req.kv.req_pool_idx, :n]

        layer_ids = (
            range(self.mem_pool_device.start_layer, self.mem_pool_device.end_layer)
            if self.is_m3_hisparse
            else range(self.mem_pool_device.layer_num)
        )
        for layer_id in layer_ids:
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_indices,
                device_locs,
                layer_id,
                io_backend="kernel",
            )

    def alloc_device_buffer(self, req: Req) -> None:
        if self.is_dsv4_hisparse:
            allocated_len = req.extend_end
            alloc_size = self.padded_buffer_size
        else:
            allocated_len = req.kv.kv_allocated_len
            page_size = self.mem_pool_device.page_size
            # Allocate only enough for current tokens (page-aligned).
            # When prefill already fills device_buffer_size, include the reserved page.
            alloc_size = min(
                ((allocated_len + page_size - 1) // page_size) * page_size,
                self.device_buffer_size,
            )
            # Spec verify needs the complete, graph-stable buffer even for short prompts.
            if self.spec_swap.enabled or alloc_size == self.device_buffer_size:
                alloc_size = self.padded_buffer_size

        compressed_logical_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :allocated_len]
            )
        )
        compressed_len = len(compressed_logical_indices)

        # A PD transfer kept the slots this buffer adds since its pre-allocation.
        self._give_back_buffer_hold(req.kv.req_pool_idx)
        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(
            compressed_logical_indices, alloc_size
        )
        if buffer_indices is None:
            logger.error(
                "HiSparse: alloc_device_buffer failed for req %s "
                "(compressed_len=%d, alloc_size=%d)",
                req.rid,
                compressed_len,
                alloc_size,
            )
            raise RuntimeError("HiSparse alloc_device_buffer returned None")

        buffer_indices = buffer_indices.to(torch.int32)
        self.req_to_device_buffer[req.kv.req_pool_idx, :alloc_size] = buffer_indices
        self.req_device_buffer_size[req.kv.req_pool_idx] = alloc_size

        self.req_device_buffer_tokens[
            :, req.kv.req_pool_idx, : self.device_buffer_size
        ] = self._device_buffer_arange_i32
        self.req_device_buffer_token_locs[:, req.kv.req_pool_idx, :alloc_size] = (
            buffer_indices[:alloc_size]
        )
        self.spec_swap.allocate_scratch(req.kv.req_pool_idx)
        self.spec_swap.reset(req.kv.req_pool_idx)

    def _grow_device_buffers(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Grow device buffers for requests whose sequence length exceeds current capacity."""
        current_caps = self.req_device_buffer_size[req_pool_indices_cpu]
        short_reqs_cpu = seq_lens_cpu <= self.device_buffer_size
        needs_grow_cpu = short_reqs_cpu & (seq_lens_cpu > current_caps)

        if torch.any(needs_grow_cpu):
            page_size = self.mem_pool_device.page_size
            grow_indices = torch.where(needs_grow_cpu)[0]

            # Compute all grow sizes on CPU, then do a single bulk allocation
            req_idxs = []
            old_caps = []
            new_caps = []
            grow_sizes = []
            total_grow = 0
            for i in grow_indices.tolist():
                req_idx = int(req_pool_indices_cpu[i])
                current_cap = int(current_caps[i])
                seq_len = int(seq_lens_cpu[i])

                new_cap = min(
                    ((seq_len + page_size - 1) // page_size) * page_size,
                    self.device_buffer_size,
                )
                if new_cap == self.device_buffer_size:
                    new_cap = self.padded_buffer_size
                grow_size = new_cap - current_cap
                if grow_size <= 0:
                    continue
                req_idxs.append(req_idx)
                old_caps.append(current_cap)
                new_caps.append(new_cap)
                grow_sizes.append(grow_size)
                total_grow += grow_size

            if total_grow > 0:
                if self._resident:  # move resident pages to host to make room
                    self.reclaim_device(total_grow)
                if (
                    self.hs_el is not None
                ):  # buffers live below the band, their growth too
                    all_new_indices = (
                        self.token_to_kv_pool_allocator._hs_el_alloc_below_or_evict(
                            total_grow
                        )
                    )
                else:
                    all_new_indices = (
                        self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
                            total_grow
                        )
                    )
                if all_new_indices is None:
                    logger.error(
                        "HiSparse: _grow_device_buffers bulk alloc failed "
                        "(total_grow=%d)",
                        total_grow,
                    )
                    raise RuntimeError(
                        f"HiSparse _grow_device_buffers failed (total_grow={total_grow})"
                    )

                offset = 0
                for req_idx, current_cap, new_cap, grow_size in zip(
                    req_idxs, old_caps, new_caps, grow_sizes
                ):
                    chunk = all_new_indices[offset : offset + grow_size]
                    offset += grow_size
                    self.req_to_device_buffer[req_idx, current_cap:new_cap] = chunk
                    self.req_device_buffer_token_locs[
                        :, req_idx, current_cap:new_cap
                    ] = chunk
                    self.req_device_buffer_size[req_idx] = new_cap

        reserved_positions = (seq_lens - 1).clamp(max=self.device_buffer_size)
        return self.req_to_device_buffer[req_pool_indices, reserved_positions]

    def request_lists(self) -> Dict[ListKind, ReqList]:
        return {ListKind.HISPARSE_STAGING: self.ack_staging_queue}

    # ---- all-resident graph ----------------------------------------------------------------------------------------------------
    def _hs_build_dev_page_table(
        self, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
    ) -> None:
        """Records whether the batch just prepared is all-resident (the graph choice). The device page table itself is
        built inside the graph from the forward's own page table (resident_page_table), so nothing host-side is
        shared with the step still running."""
        rpi = tuple(int(r) for r in req_pool_indices_cpu.tolist())
        self._hs_pt_batch = None
        if not rpi or any(r not in self._resident for r in rpi):
            return
        self._hs_pt_batch = rpi
        self._hs_pt_seq = tuple(int(x) for x in seq_lens_cpu.tolist())

    def resident_page_table(self, attn_metadata) -> Optional[torch.Tensor]:
        """While the all-resident decode graph is captured: the page table its top-k
        maps selected positions through. None in every other forward.

        Built inside the graph from the forward's own page table and the allocator's
        map, by the first indexer layer of the forward; the others reuse it."""
        if (
            not self.enable_resident
            or get_capture_attention_variant() != HISPARSE_RESIDENT
        ):
            return None
        page_table = attn_metadata.real_page_table
        # A graph's warm-up and its capture each build their own.
        key = (
            id(attn_metadata),
            torch.cuda.is_current_stream_capturing(),
            page_table.data_ptr(),
            tuple(page_table.shape),
        )
        memo = self._hs_pt_memo
        if memo is not None and memo[0] == key:
            return memo[1]
        table = resident_page_table(
            page_table=page_table,
            mapping=self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping,
            seq_lens=attn_metadata.dsa_seqlens_expanded,
            page_size=self.page_size,
        )
        self._hs_pt_memo = (key, table)
        return table

    def hs_batch_all_resident(self, forward_batch) -> bool:
        """The graph-variant choice: every request resident, and the page table built for exactly this batch."""
        ok = False
        if self.enable_resident and forward_batch.forward_mode.is_target_verify():
            # the verify graph is chosen right after this round's preparation, in the same thread
            ok = self._hs_verify_sel is not None and self._hs_verify_sel == int(
                forward_batch.batch_size
            )
        elif self._hs_pt_batch is not None and self.enable_resident:
            rpi_cpu = forward_batch.req_pool_indices_cpu
            seq_cpu = forward_batch.seq_lens_cpu
            if rpi_cpu is not None:
                ok = tuple(int(r) for r in rpi_cpu.tolist()) == self._hs_pt_batch
            elif (
                seq_cpu is not None
            ):  # plain decode keeps this host mirror; same lengths in the same order
                ok = (
                    int(forward_batch.batch_size) == len(self._hs_pt_batch)
                    and tuple(int(x) for x in seq_cpu.tolist()) == self._hs_pt_seq
                )
        self._hs_sel[ok] += 1
        if sum(self._hs_sel) % 500 == 0:
            logger.info(
                "HiSparse graph variants: %d hs_resident, %d hs_mixed replays so far",
                self._hs_sel[1],
                self._hs_sel[0],
            )
        return ok

    # ---- residency ----------------------------------------------------------------------------------------------
    def _gpu_rows(self, rows) -> torch.Tensor:
        # Pinned, non-blocking H2D: a plain torch.tensor(..., device=) would wait for the schedule stream's queued work.
        return torch.tensor(rows, dtype=torch.int64, pin_memory=True).to(
            self.device, non_blocking=True
        )

    def _can_stay_resident(self, req: Req) -> bool:
        # Short requests already live wholly in their buffer (the kernel's short fast path).
        return (
            self.enable_resident
            and self.host_token_len(req.kv.kv_allocated_len) > self.device_buffer_size
        )

    def _make_resident(self, req: Req) -> None:
        rid = req.kv.req_pool_idx
        self._resident[rid] = req
        self.req_resident[rid] = 1
        self.req_device_buffer_size[rid] = 0
        if self.hs_spec_res:  # positions [0, staged) have their prefill device slots
            self._hs_dev_len[rid] = self._hs_staged_len[rid]
        if self.hs_exclusive:  # whole pages, or nothing: _exclusive_staged_len
            self.paging.host_len[rid] = int(self.req_to_host_pool_allocated_len[rid])
        # The staged prefix's whole pages may leave the device pool from now on.
        self.paging.registered_len[rid] = 0
        self.paging.register(rid, req)

    def reclaimable_slots(self) -> int:
        """Device slots that moving every page that may leave the device pool to
        host would free: the allocator counts them as available for admission."""
        P = self.page_size
        seal = sum(
            max(self.paging.sealable_end(r) - self.paging.registered_len.get(rid, 0), 0)
            // P
            for rid, r in self._resident.items()
        )
        # Buffered requests grow their buffers page by page up to the full
        # buffer: keep that room.
        sizes = self.req_device_buffer_size
        grow = int((self.padded_buffer_size - sizes[sizes > 0]).sum())
        return max((self.paging.num_evictable_pages + seal) * P - grow, 0)

    def _exclusive_reach(self, device_free: int) -> tuple[int, int]:
        """Exclusive placement: tokens that can still get a device slot, and tokens
        that can still get a host slot. A slot on one side is made by moving a
        page to the other, so each side counts what it can reach."""
        P = self.page_size
        paging = self.paging
        host_free = self.mem_pool_host.available_size()
        dual = paging.num_evictable_pages * P  # a frame and a host slot: either can go
        exclusive = (paging.num_exclusive_pages + paging.untracked_sealable_pages()) * P
        on_host = paging._n_onhost * P
        # Buffered requests grow their buffers page by page up to the full
        # buffer: keep that room.
        sizes = self.req_device_buffer_size
        grow = int((self.padded_buffer_size - sizes[sizes > 0]).sum())
        # Device slots: free ones, those of pages with a host copy, and those of
        # pages that can still get one.
        device_reach = device_free + max(dual + min(exclusive, host_free) - grow, 0)
        # Host slots: free ones, those of pages that also sit in a frame, and those
        # of pages that can move into a free frame.
        host_reach = host_free + dual + min(on_host, max(device_free, 0))
        return device_reach, host_reach

    def exclusive_available_slots(self, *, logical_free: int, device_free: int) -> int:
        """Exclusive placement: tokens the pools can still take. A token is written
        to a device slot, so this is the device side; what needs host slots (a
        buffered request) is charged apart: `exclusive_host_shortfall` at
        admission, `exclusive_host_fits` before a decode step."""
        return min(logical_free, self._exclusive_reach(device_free)[0])

    def exclusive_host_shortfall(self, *, device_free: int) -> int:
        """How far the host side is behind the device side: admission adds it to a
        buffered prompt's charge, which then waits for host room."""
        device_reach, host_reach = self._exclusive_reach(device_free)
        return max(device_reach - host_reach, 0)

    def exclusive_host_fits(self, requests, *, device_free: int) -> bool:
        """Whether the buffered requests of a decode step can each get a host page."""
        num_buffered = sum(
            1 for req in requests if req.kv.req_pool_idx not in self._resident
        )
        if num_buffered == 0:
            return True
        return num_buffered * self.page_size <= self._exclusive_reach(device_free)[1]

    def reserve_buffered_host_rows(self, reqs) -> None:
        """Exclusive placement with a host pool below the logical pool: a prompt
        that will stay buffered takes its host rows with its request slot, as
        admission charged them (nothing else can take them before staging)."""
        if not self.hs_host_scarce:
            return
        P = self.page_size
        for req in reqs:
            rid = req.kv.req_pool_idx
            total = len(req.full_untruncated_fill_ids)
            held = int(self.req_to_host_pool_allocated_len[rid])
            if total > self.device_buffer_size or held >= total:
                continue
            self.paging.ensure_host(-(-total // P) * P - held, must=True)
            self.mem_pool_host.alloc_paged_token_slots(
                self.req_to_host_pool,
                self.req_to_host_pool_allocated_len,
                rid,
                held,
                total - held,
            )

    def forget_spare_scratch(self) -> None:
        self.spec_swap.forget_spare_scratch()

    def spare_scratch_slots(self) -> int:
        return self.spec_swap.spare_scratch_slots()

    def frame_reuse_fences(self) -> list:
        """Events a writer that follows no stream (a PD prefill) waits for before
        it writes device frames handed out so far: a launched forward may still
        read a frame through the page table it was launched with."""
        if not self.enable_resident or self.decode_producer_stream is None:
            return []
        launched = device_module.Event()
        launched.record(self.decode_producer_stream)
        return [launched]

    def room_without_copy(self, need_slots: int) -> bool:
        """Paged residency: whether an arrival can take `need_slots` device slots now
        without the loop waiting for a copy to host or for an ended request's last
        forward. False: what is missing is on its way; ask again on a later pass."""
        if not self.enable_resident:
            return True
        self.run_deferred_releases()
        dev = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        if dev.available_size() < need_slots:
            self.spec_swap.free_spare_scratch()
        if dev.available_size() >= need_slots:
            return True
        if self._release_queue and self.decode_producer_stream is not None:
            return False
        return self.paging.room_without_copy(need_slots)

    def note_frame_wait(self, rid: str, waited: bool) -> None:
        """Arrivals that waited for device frames: how long, from the first refusal
        to the pre-allocation."""
        now = time.perf_counter()
        if waited:
            if len(self._frame_wait_since) >= 4096:  # requests that left while waiting
                self._frame_wait_since.clear()
            self._frame_wait_since.setdefault(rid, now)
            return
        since = self._frame_wait_since.pop(rid, None)
        if since is None:
            return
        waits = self._frame_waits  # the last 1,000
        waits.append((now - since) * 1e3)
        self._frame_wait_count += 1
        if self._frame_wait_count % 50 == 0:
            ordered = sorted(waits)
            logger.info(
                "HiSparse paged: %d arrivals waited for device frames: mean %.1f ms, "
                "p50 %.1f, p90 %.1f, max %.1f",
                self._frame_wait_count,
                sum(waits) / len(waits),
                ordered[len(waits) // 2],
                ordered[len(waits) * 9 // 10],
                ordered[-1],
            )
            del waits[:-950]

    def reclaim_device(self, need_slots: int) -> bool:
        """Move cold pages to host until `need_slots` device slots are free."""
        # An ended request's frames are free once its last forward has run: wait
        # for that before moving anything else.
        self.run_deferred_releases(block=True)
        self.spec_swap.free_spare_scratch()
        return self.paging.reclaim(need_slots)

    # ---- elastic range ------------------------------------------------------------------------------------------------
    def _hs_el_budget(self):
        el = self.hs_el
        kv, ik = el["kv"], el["ik"]
        band = (kv.mapped - kv.perm) * kv.layers
        return el["E"] - band - ik.mapped * ik.layers

    def _hs_el_sync(self) -> None:
        # nothing may touch memory being unmapped or moved: the in-flight batch, the staging and backup copies
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()
        torch.cuda.synchronize()
        if self.enable_resident:
            self.paging.finish_seal(block=True)  # its frames may move

    def hs_el_evict_below_band(self) -> bool:
        """Make room below the band: cold pages there leave the device pool."""
        self.run_deferred_releases(block=True)
        if self.spec_swap.free_spare_scratch():  # kept scratch sits below the band
            return True
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()
        floor = (self.hs_el["k_min"] + self.page_size) // self.page_size
        n = self.paging.evict(self.paging.evict_batch_pages, hi_frame=floor)
        if n == 0 and self.hs_exclusive:
            self.paging.track_all()
            batch = self.paging.evict_batch_pages
            if self.paging.seal_exclusive(batch, hi_frame=floor) == 0:
                # Nothing below can leave for host: exclusive pages move up into the band.
                self.hs_el_stats[4] += 1
                return self.paging.lift(batch, floor) > 0
            n = self.paging.evict(batch, hi_frame=floor)
        elif n == 0:
            self.paging.seal_all()
            n = self.paging.evict(self.paging.evict_batch_pages, hi_frame=floor)
        self.hs_el_stats[4] += 1
        return n > 0

    def _hs_el_log_high(self) -> int:
        """Highest allocated logical page (0 when none): the logical allocator's free list is sorted below its limit."""
        el = self.hs_el
        alloc = self.token_to_kv_pool_allocator.logical_attn_allocator
        alloc.merge_and_sort_free()
        lim = el["log_max_page"]
        fp = alloc.free_pages
        n = int(fp.numel())
        if n == 0:
            return lim
        # free pages form [1..lim] minus used; the top run of free pages ends at lim
        tail = fp[-min(n, lim) :]
        expect = torch.arange(lim - tail.numel() + 1, lim + 1, device=fp.device)
        same = (tail == expect).flip(0).to(torch.int64).cumprod(0).sum()
        return lim - int(same)

    def hs_el_grow_band(self, log: bool = False, keep_extra: int = 0) -> None:
        """Lend index-K mapped above the logical high-water + headroom (+ keep_extra pages a pending logical
        allocation will take) to the band."""
        el = self.hs_el
        alloc = self.token_to_kv_pool_allocator
        kv, ik, pb, row, P = (
            el["kv"],
            el["ik"],
            el["ik_page_bytes"],
            el["row"],
            self.page_size,
        )
        if el["pinned"]:
            # A PD prefill writes these ranges: mapped once, nothing to lend.
            if log:
                logger.info(
                    "HiSparse elastic: ranges pinned for NVLink transfers (no band; "
                    "index keys mapped for %d logical pages)",
                    ik.mapped // pb,
                )
            return
        keep_pages = self._hs_el_log_high() + el["headroom"] + 2 + keep_extra
        keep = max(0, min(ik.stride, (keep_pages + 1) * pb))
        g = ik.g
        if -(-keep // g) * g < ik.mapped:
            self._hs_el_sync()
            new_max = min(
                alloc.logical_attn_allocator.num_pages, -(-keep // g) * g // pb - 1
            )
            alloc.hs_el_set_limit("log", new_max)
            ik.shrink_to(keep)
        # map band chunks while the budget allows
        added = 0
        while self._hs_el_budget() >= kv.chunk_bytes() and kv.mapped + kv.g <= kv.cap:
            kv.grow_to(kv.mapped + kv.g)
            added += 1
        if added:
            alloc.hs_el_set_limit("dev", kv.mapped // row // P - 1)
            self.hs_el_stats[0] += 1
        if added or log:
            logger.info(
                "HiSparse elastic: band %d slots (%.1f GiB), index-K mapped for %d logical pages; budget left %.2f "
                "GiB; stats %s; runtime cuMemCreate %d",
                kv.mapped // row - el["k_min"] - P,
                (kv.mapped - kv.perm) * kv.layers / 2**30,
                ik.mapped // pb,
                self._hs_el_budget() / 2**30,
                self.hs_el_stats,
                int(kv.x.creates()) - el.get("creates0", 0),
            )

    def hs_el_ensure_logical(self, n_pages: int) -> None:
        """Index-K for n_pages more logical pages (+ headroom); takes the band back when the budget is short."""
        el = self.hs_el
        alloc = self.token_to_kv_pool_allocator
        la = alloc.logical_attn_allocator
        if la.available_size() // self.page_size >= n_pages + 1:
            return
        kv, ik, pb, row, P = (
            el["kv"],
            el["ik"],
            el["ik_page_bytes"],
            el["row"],
            self.page_size,
        )
        if el["log_max_page"] >= la.num_pages:
            return  # all logical pages mapped already
        want = min(ik.stride, (el["log_max_page"] + 2 + n_pages + el["headroom"]) * pb)
        need = (-(-want // ik.g) * ik.g - ik.mapped) * ik.layers
        if need > self._hs_el_budget():
            short = need - self._hs_el_budget()
            chunks = -(-short // kv.chunk_bytes())
            self._hs_el_shrink_band(max(kv.perm, kv.mapped - chunks * kv.g))
        ik.grow_to(want)
        alloc.hs_el_set_limit("log", min(la.num_pages, ik.mapped // pb - 1))

    def _hs_el_shrink_band(self, new_bytes: int) -> None:
        """Take band rows at and above new_bytes back: mapped pages there move below (fresh pages), evicting cold
        pages when there is no room; then the chunks are unmapped."""
        el = self.hs_el
        alloc = self.token_to_kv_pool_allocator
        dev = alloc.hisparse_attn_allocator
        kv, row, P = el["kv"], el["row"], self.page_size
        new_rows = new_bytes // row
        new_max_page = new_rows // P - 1
        self._hs_el_sync()
        alloc.hs_el_set_limit("dev", new_max_page)  # free pages above are dropped
        lo = (new_max_page + 1) * P
        mapping = alloc.full_to_hisparse_device_index_mapping
        while True:
            # Sealed pages above the new limit leave the device pool without a
            # copy. Every pass: making room below (hs_el_evict_below_band) may seal
            # pages that sit above the limit, and a relocated sealed page would
            # keep its frame registered above it.
            self.paging.evict(self.paging.num_device_pages, lo_frame=new_max_page + 1)
            # Map entries that name host rows are not band pages.
            sel = torch.nonzero(
                (mapping[:-1] >= lo) & (mapping[:-1] < self.paging.host_row0)
            ).squeeze(1)
            if sel.numel() == 0:
                break
            slots = mapping[sel]
            pages = torch.unique(slots // P)
            new = dev.alloc(len(pages) * P)
            if new is None:
                if not self.hs_el_evict_below_band() and not (
                    # Exclusive pages above the limit can leave for host instead.
                    self.hs_exclusive
                    and self.paging.seal_exclusive(
                        len(pages), lo_frame=new_max_page + 1
                    )
                ):
                    raise RuntimeError(
                        "HiSparse elastic: no room to take the band back"
                    )
                self._hs_el_sync()
                continue
            new_pages = new.view(-1, P)[:, 0] // P
            src = (pages[:, None] * P + torch.arange(P, device=self.device)).reshape(-1)
            dst = (
                new_pages[:, None] * P + torch.arange(P, device=self.device)
            ).reshape(-1)
            alloc.hs_copy_rows(src, dst)
            idx = torch.searchsorted(pages, slots // P)
            mapping[sel] = new_pages[idx] * P + slots % P
            if self.hs_exclusive:  # registered pages without a host slot move too
                self.paging.xstats[4] += int(self.paging.xframe[pages].sum())
                self.paging.relocate(pages, new_pages)
            self.hs_el_stats[2] += len(pages)
            break
        # Nothing else may hold pages above the limit: buffers and scratch live below
        # the band. Rows of buffered requests only: plain decode writes a resident
        # row's newest-token slot into the reserved column of
        # req_device_buffer_token_locs, where nothing reads it.
        sw = self.spec_swap
        buf_rows = (
            torch.nonzero(self.req_device_buffer_size > 0).squeeze(1).to(self.device)
        )
        held = int((self.req_to_device_buffer[buf_rows] >= lo).sum()) + int(
            (self.req_device_buffer_token_locs[:, buf_rows] >= lo).sum()
        )
        if sw.enabled:
            held += int((sw.req_to_scratch >= lo).sum())
        if held:
            raise RuntimeError(
                f"HiSparse elastic: {held} buffer/scratch slots above the band limit {lo}"
            )
        # No free page above the limit either (evictions above may have freed some; _release_page_ids drops them).
        fp = torch.cat((dev.free_pages, *dev.staged_pages))
        if fp.numel() and int(fp.max()) > new_max_page:
            raise RuntimeError(
                f"HiSparse elastic: free device page {int(fp.max())} above the band limit {new_max_page}"
            )
        torch.cuda.synchronize()
        self._hs_pt_memo = None
        kv.shrink_to(new_bytes)
        self.hs_el_stats[1] += 1
        logger.info(
            "HiSparse elastic: band taken back to %d slots; stats %s; runtime cuMemCreate %d",
            new_rows - el["k_min"] - P,
            self.hs_el_stats,
            int(kv.x.creates()) - el.get("creates0", 0),
        )

    def _hs_spec_res_prepare(self, batch) -> List[int]:
        """give every resident request of this verify batch the device pages of the logical pages it holds
        (kv_allocated_len, reserved ahead by spec decode), mirrored page for page: first the rest of the device page that
        holds position L - 1 (L = positions already mapped), then whole new pages. Demotes resident requests (largest
        first) when the free pages do not cover it. Returns the batch rows that are resident afterwards."""
        P = self.page_size
        if self.enable_resident:
            self.paging.step()
        if self.hs_el is not None:  # lend index-K slack to the band now and then
            self.hs_el_stats[5] += 1
            if self.hs_el_stats[5] % 200 == 0:
                self.hs_el_grow_band()
        res = self._resident
        dev = self.token_to_kv_pool_allocator.hisparse_attn_allocator

        def plan(rows):
            out = []
            for i in rows:
                req = batch.reqs[i]
                rid = req.kv.req_pool_idx
                L, E = self._hs_dev_len[rid], int(req.kv.kv_allocated_len)
                if E > L:
                    out.append((rid, L, min(E, -(-L // P) * P), E))
            return out

        rows = [i for i, req in enumerate(batch.reqs) if req.kv.req_pool_idx in res]
        todo = plan(rows)
        need = sum(E - pe for _, _, pe, E in todo)
        if need > dev.available_size():
            self.reclaim_device(need)
            rows = [i for i in rows if batch.reqs[i].kv.req_pool_idx in res]
            todo = plan(rows)
            need = sum(E - pe for _, _, pe, E in todo)
            if need > dev.available_size():
                raise RuntimeError(
                    f"HiSparse spec residency: {need} device slots needed after reclaim"
                )
        if todo:
            full = dev.alloc(need) if need else None
            if need and full is None:
                raise RuntimeError(
                    f"HiSparse spec residency: device allocation of {need} slots failed"
                )
            r2t = self.req_to_token_pool.req_to_token
            mapping = (
                self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
            )
            log_parts, dev_parts, off = [], [], 0
            for rid, L, pe, E in todo:
                if pe > L:  # the rest of the device page that holds position L - 1
                    # a one-element index: a scalar one reads the slot back to the host
                    base = mapping[r2t[rid, L - 1 : L].to(torch.int64)]
                    log_parts.append(r2t[rid, L:pe])
                    dev_parts.append(base + 1 + self._hs_arange_page[: pe - L])
                if E > pe:
                    log_parts.append(r2t[rid, pe:E])
                    dev_parts.append(full[off : off + E - pe].to(torch.int64))
                    off += E - pe
                self._hs_dev_len[rid] = E
            mapping[torch.cat(log_parts).to(torch.int64)] = torch.cat(dev_parts)
        st = self._hs_spec_steps
        st[0] += 1
        st[1] += len(rows)
        st[2] += len(batch.reqs) - len(rows)
        if st[0] % 1000 == 0:
            logger.info(
                "HiSparse spec resident: %d verify steps, %.1f resident / %.1f buffered "
                "rows per step; %d resident now; reclaims before scratch %d, before "
                "buffer %d",
                st[0],
                st[1] / st[0],
                st[2] / st[0],
                len(res),
                self.hs_spec_reclaims[0],
                self.hs_spec_reclaims[1],
            )
        return rows

    def _hs_lean_decode(
        self,
        seq_lens,
        out_cache_loc,
        req_pool_indices,
        seq_lens_cpu,
        req_pool_indices_cpu,
    ) -> bool:
        """decode prep of a batch whose requests are all resident: what _resident_decode_locs does for such a
        batch (each new token's paged device slot next to the previous token's, or a fresh page), mapped from its
        logical slot. Returns False, having done nothing, for any other batch or when the free device pages do not
        cover the step (the general path reclaims)."""
        rpi = req_pool_indices_cpu.tolist()
        res = self._resident
        if not rpi or any(r not in res for r in rpi):
            self._hs_lean[1] += 1
            return False
        page = self.mem_pool_device.page_size
        dev = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        need_pages = sum(1 for s in seq_lens_cpu.tolist() if (s - 1) % page == 0)
        if need_pages * page > dev.available_size():
            if not self.reclaim_device(need_pages * page):
                self._hs_lean[1] += 1
                return False
        self._hs_lean[0] += 1
        self._resident_steps += 1
        if self._resident_steps % 1000 == 0:
            logger.info(
                "HiSparse resident: %d of %d running requests resident; decode steps "
                "prepared for all-resident batches %d, for mixed batches %d",
                len(rpi),
                len(rpi),
                self._hs_lean[0],
                self._hs_lean[1],
            )
        mapping = self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
        prev_logical = self.req_to_token_pool.req_to_token[
            req_pool_indices, seq_lens - 2
        ].to(torch.int64)
        new_locs = dev.alloc_decode(seq_lens, seq_lens_cpu, mapping[prev_logical])
        if new_locs is None:
            raise RuntimeError("HiSparse lean decode allocation failed")
        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
            out_cache_loc
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            new_locs
        )
        if self.enable_resident:
            self._hs_build_dev_page_table(
                req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )
        return True

    def _resident_decode_locs(
        self,
        reserved_buffer_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Device slots for this step's new tokens of resident requests (paged, next to the previous token)."""
        rpi_cpu = req_pool_indices_cpu.tolist()
        rows = [i for i, r in enumerate(rpi_cpu) if r in self._resident]
        self._resident_steps += 1
        if self._resident_steps % 1000 == 0:
            logger.info(
                "HiSparse resident: %d of %d running requests resident",
                len(rows),
                len(rpi_cpu),
            )
        if not rows:
            return reserved_buffer_loc
        page = self.mem_pool_device.page_size
        dev = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        need_pages = sum(1 for i in rows if (int(seq_lens_cpu[i]) - 1) % page == 0)
        if need_pages * page > dev.available_size():
            self.reclaim_device(need_pages * page)
        idx = self._gpu_rows(rows)
        rr = req_pool_indices[idx]
        sl = seq_lens[idx]
        prev_logical = self.req_to_token_pool.req_to_token[rr, sl - 2].to(torch.int64)
        prev_dev = (
            self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping[
                prev_logical
            ]
        )
        new_locs = dev.alloc_decode(sl, seq_lens_cpu[rows], prev_dev)
        if new_locs is None:
            raise RuntimeError(
                "HiSparse resident decode allocation failed after reclaim"
            )
        reserved_buffer_loc[idx] = new_locs.to(reserved_buffer_loc.dtype)
        return reserved_buffer_loc

    def move_ready_reqs(self, dst: ReqList) -> List[Req]:
        """Move the staged requests whose host copy has landed into `dst`, and
        return them."""
        self.run_deferred_releases()
        ready_reqs: List[Req] = []
        if len(self.ack_staging_queue) == 0:
            return ready_reqs

        staged = list(self.ack_staging_queue.items())
        finish_count = 0
        for _, finish_event, _ in staged:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make sure the same update to scheduler
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())
        move([req for _, _, req in staged[:finish_count]], dst)
        for _, _, req in staged[:finish_count]:
            # prepare device buffer and update req, or keep the whole prefill resident
            if self._can_stay_resident(req):
                self._make_resident(req)
            else:
                if (
                    self._resident
                ):  # a prompt just under the buffer size takes one extra page
                    self.reclaim_device(self.mem_pool_device.page_size)
                self.alloc_device_buffer(req)
            self._skip_first_backup[req.kv.req_pool_idx] = True
            req.hisparse_staging = False
            ready_reqs.append(req)
        return ready_reqs

    def prepare_spec_verify(self, batch: ScheduleBatch) -> None:
        self.spec_swap.prepare_verify(batch)

    def commit_spec_accept_tokens(
        self, batch: ScheduleBatch, accept_indices: torch.Tensor
    ) -> None:
        self.spec_swap.commit_accept_tokens(batch, accept_indices)

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        self._eager_backup_previous_token(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )
        if self.enable_resident:
            self.paging.step()

        if not self.is_dsv4_hisparse:
            if self.enable_resident and self._hs_lean_decode(
                seq_lens,
                out_cache_loc,
                req_pool_indices,
                seq_lens_cpu,
                req_pool_indices_cpu,
            ):
                return
            # Grow device buffers if needed and resolve the latest-token slot.
            reserved_buffer_loc = self._grow_device_buffers(
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )
            if (
                self._resident
            ):  # resident requests' new tokens get their own device slots
                reserved_buffer_loc = self._resident_decode_locs(
                    reserved_buffer_loc,
                    seq_lens,
                    req_pool_indices,
                    seq_lens_cpu,
                    req_pool_indices_cpu,
                )
            self.req_device_buffer_token_locs[
                :, req_pool_indices, self.device_buffer_size
            ] = reserved_buffer_loc.to(torch.int32)

            compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
                out_cache_loc
            )
            # Page-size-one allocation creates a temporary slot before remapping
            # the new token into the request's reserved device-buffer slot.
            if _is_hip or self.mem_pool_device.page_size == 1:
                previous_locs = self.mem_pool_device._translate_loc_to_hisparse_device(
                    compressed_locs
                )
                stale_locs = previous_locs[
                    (previous_locs > 0) & (previous_locs != reserved_buffer_loc)
                ]
                if stale_locs.numel() > 0:
                    self.token_to_kv_pool_allocator.free_hisparse_indices(stale_locs)

            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = reserved_buffer_loc
            if self.enable_resident:
                self._hs_build_dev_page_table(
                    req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
                )
            return

        active_reqs = seq_lens % self.compress_ratio == 0
        if not torch.any(active_reqs):
            return

        active_seq_lens = seq_lens[active_reqs]
        active_out_cache_loc = out_cache_loc[active_reqs]
        active_req_pool_indices = req_pool_indices[active_reqs]

        compressed_seq_lens = active_seq_lens // self.compress_ratio
        reserved_positions = (compressed_seq_lens - 1).clamp(
            max=self.device_buffer_size
        )
        reserved_buffer_loc = self.req_to_device_buffer[
            active_req_pool_indices, reserved_positions
        ]

        self.req_device_buffer_token_locs[
            :, active_req_pool_indices, self.device_buffer_size
        ] = reserved_buffer_loc.to(torch.int32)

        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
            active_out_cache_loc
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            reserved_buffer_loc
        )

    def _eager_backup_previous_token(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Back up the previous compressed token to host memory.

        Each newly produced compressed token (one per `compress_ratio` decode
        steps) must be backed up to host so the swap-in kernel can later
        recover it.

        Two cases are skipped:
        - The first decode step right after staging: all prefill tokens were
          already backed up during staging, so there is nothing new to save.
        - Steps where `(seq_len - 1) % compress_ratio != 0`: no new compressed
          token was produced this step.
        """
        # Build the list of batch positions that need a host backup.
        # Skip the first decode step after staging (prefill already backed up),
        # and skip non-aligned steps that did not produce a new compressed token.
        backup_indices = []
        for i in range(len(seq_lens_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            if self._skip_first_backup[req_idx]:
                self._skip_first_backup[req_idx] = False
                continue
            # A resident request's tokens are copied to host when their page
            # leaves the device pool, not step by step.
            if (
                int(seq_lens_cpu[i]) - 1
            ) % self.compress_ratio == 0 and req_idx not in self._resident:
                backup_indices.append(i)

        if not backup_indices:
            return

        # Use a pinned, non-blocking H2D copy to avoid synchronizing the host
        # with pending work on the schedule stream.
        backup_indices_gpu = torch.tensor(
            backup_indices, dtype=torch.int64, device="cpu", pin_memory=True
        ).to(self.device, non_blocking=True)
        backup_req_indices = req_pool_indices[backup_indices_gpu]

        # The previous compressed token's position and its device buffer slot:
        #  compressed_pos = (seq_len - 1) // compress_ratio - 1
        #  - short: slot = compressed_pos          (within the regular buffer)
        #  - long:  slot = device_buffer_size      (the reserved slot)
        prev_seq_lens = seq_lens[backup_indices_gpu] - 1
        compressed_prev_seq_lens = prev_seq_lens // self.compress_ratio
        actual_compressed_pos = compressed_prev_seq_lens - 1

        buffer_slot = actual_compressed_pos.clamp(max=self.device_buffer_size)

        device_locs = self.req_to_device_buffer[backup_req_indices, buffer_slot]

        if self.hs_exclusive:
            self.paging.ensure_host(
                sum(
                    self.page_size
                    for i in backup_indices
                    if (int(seq_lens_cpu[i]) - 1) // self.compress_ratio
                    > int(
                        self.req_to_host_pool_allocated_len[
                            int(req_pool_indices_cpu[i])
                        ]
                    )
                ),
                must=True,
            )
        host_locs_list = []
        for i in backup_indices:
            req_idx = int(req_pool_indices_cpu[i])
            start_pos = (int(seq_lens_cpu[i]) - 1) // self.compress_ratio - 1
            host_locs = self.mem_pool_host.alloc_paged_token_slots(
                self.req_to_host_pool,
                self.req_to_host_pool_allocated_len,
                req_idx,
                start_pos,
                1,
            )
            host_locs_list.append(host_locs)
        host_locs = torch.cat(host_locs_list)

        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            if self.is_dsv4_hisparse:
                # One scattered token per request: a multiple of the page size
                # must not be mistaken for whole pages.
                self.mem_pool_host.backup_tokens_from_device_all_layer(
                    host_locs, device_locs
                )
            else:
                self.mem_pool_host.backup_from_device_all_layer(
                    self.mem_pool_device,
                    host_locs,
                    device_locs,
                    io_backend="kernel",
                )
            self._backup_done_event.record()
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if backup_req_indices.is_cuda:
                backup_req_indices.record_stream(self.decode_backup_stream)
            if actual_compressed_pos.is_cuda:
                actual_compressed_pos.record_stream(self.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(self.decode_backup_stream)
        self._has_pending_backup = True

    def wait_for_pending_backup(self) -> None:
        if not self._has_pending_backup:
            return
        self._backup_done_event.wait(device_module.current_stream())
        self._has_pending_backup = False

    def naive_load_topk(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_tokens: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Load top-k selected tokens into device memory and return their device indices.

        This is a naive per-request loop implementation for debugging/validation.
        Production code uses swap_in_selected_pages (JIT CUDA kernel) instead.

        Note: dsv4 hisparse is not supported — DeepSeekV4SingleKVPoolHost has no
        load_to_device_per_layer and indices live in compressed space. Currently
        only used as a kernel oracle in test_hisparse_unit.py (non-dsv4 path).

        Args:
            req_pool_indices: Pool indices for each request.  Shape: (num_reqs,)
            seq_lens: Sequence lengths for each request.  Shape: (num_reqs,)
            top_k_tokens: Selected token positions per request.  Shape: (num_reqs, top_k)
            layer_id: The layer to load KV cache for.

        Returns:
            Device KV cache indices for the selected tokens.  Shape: (num_reqs, top_k)
        """
        assert not self.is_dsv4_hisparse, (
            "naive_load_topk is not implemented for dsv4 hisparse"
        )
        num_reqs = req_pool_indices.size(0)
        top_k_indices = torch.full(
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )

        for i in range(num_reqs):
            seq_len = int(seq_lens[i].item())
            top_n = min(seq_len, self.top_k)
            if top_n == 0:
                continue

            req_idx = int(req_pool_indices[i].item())
            selected_tokens = top_k_tokens[i, :top_n].to(dtype=torch.int64)

            assert torch.all(selected_tokens >= 0), (
                f"Req {req_idx}: selected tokens contain negative positions"
            )
            assert torch.all(selected_tokens < seq_len), (
                f"Req {req_idx}: selected tokens {selected_tokens.tolist()} "
                f"out of range for seq_len={seq_len}"
            )

            if seq_len <= self.device_buffer_size:
                device_indices = self.req_to_device_buffer[req_idx, selected_tokens]
            else:
                device_indices = torch.empty(
                    top_n, dtype=torch.int64, device=self.device
                )

                is_latest_token = selected_tokens == (seq_len - 1)
                needs_host_load = ~is_latest_token

                device_indices[is_latest_token] = self.req_to_device_buffer[
                    req_idx, self.device_buffer_size
                ]

                num_to_load = int(needs_host_load.sum().item())
                if num_to_load > 0:
                    tokens_to_load = selected_tokens[needs_host_load]
                    host_locs = self.req_to_host_pool[req_idx, tokens_to_load]

                    invalid_mask = host_locs < 0
                    if torch.any(invalid_mask):
                        bad_positions = tokens_to_load[invalid_mask].tolist()
                        raise AssertionError(
                            f"Req {req_idx} (seq_len={seq_len}, layer={layer_id}): "
                            f"missing host backup at token positions {bad_positions}"
                        )

                    buffer_locs = self.req_to_device_buffer[req_idx, :num_to_load]
                    device_indices[needs_host_load] = buffer_locs

                    self.mem_pool_host.load_to_device_per_layer(
                        self.mem_pool_device,
                        host_locs,
                        buffer_locs,
                        layer_id,
                        io_backend="kernel",
                    )

            top_k_indices[i, :top_n] = device_indices.to(torch.int32)

        return top_k_indices

    def _release_staging(self, req: Req) -> None:
        """Free a staged request's host + device resources."""
        # Wait for any in-flight staging DMA to complete before freeing
        self.write_staging_stream.synchronize()
        self._give_back_buffer_hold(req.kv.req_pool_idx)

        prefill_len = req.extend_end
        allocated_locs = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :prefill_len
        ]
        self.token_to_kv_pool_allocator.free_hisparse(allocated_locs)

        # Free host memory that was allocated during admit_request_into_staging
        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.kv.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.kv.req_pool_idx],
        )
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        self.req_to_host_pool[req.kv.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.kv.req_pool_idx] = 0
        self._skip_first_backup[req.kv.req_pool_idx] = False
        self.spec_swap.free_unrotated_scratch(req.kv.req_pool_idx)
        req.hisparse_staging = False

    def defer_release(self, req: Req, release: Callable[[], None]) -> bool:
        """Paged residency under the overlap loop: give an ended request's pages
        back once the forward launched before its end has run, instead of making
        this stream (and every host read after it) wait for that forward. False:
        the caller releases now."""
        if not self.enable_resident or self.decode_producer_stream is None:
            return False
        if req.hisparse_staging:  # no forward has read its rows: released at once
            return False
        rid = req.kv.req_pool_idx
        if rid in self._release_pending:  # ended again before its release ran
            return True
        self.run_deferred_releases()
        # Everything queued so far on the streams that may still use its pages.
        forward_done = device_module.Event()
        forward_done.record(self.decode_producer_stream)
        backup_done = device_module.Event()
        backup_done.record(self.decode_backup_stream)
        # And on this stream: a request that ends in the pass it was pre-allocated in has its row, its map
        # entries and its scratch table still queued here behind the batch in flight (no launch has waited
        # for them); so are the table writes of a seal or an eviction issued in this pass.
        issued = device_module.Event()
        issued.record(device_module.current_stream())
        fences = [forward_done, backup_done, issued]
        # A seal running now may be copying its pages; none started later will.
        if rid in self._resident:
            seal_done = self.paging.seal_fence()
            if seal_done is not None:
                fences.append(seal_done)
            self.paging.unlist(rid)
        self._release_pending.add(rid)
        self._release_queue.append((req, tuple(fences), release))
        return True

    def run_deferred_releases(self, block: bool = False) -> None:
        """Release the ended requests whose last users have run, oldest first.
        block: wait for them (the pages are needed now, or the pools are reset).
        Called where the loop collects its requests, so the release overlaps the
        forward in flight, and wherever memory is short."""
        queue = self._release_queue
        while queue:  # None on a coordinator built without __init__
            req, fences, release = queue[0]
            for event in fences:
                if not event.query():
                    if not block:
                        return
                    event.synchronize()
            if self.paging is not None:
                # An eviction issued before this point picks its pages when its stream gets to it, and an
                # ended request's pages are among the candidates: the release follows it, never runs beside it.
                evicting = self.paging.evict_fence()
                if evicting is not None:
                    if not block:
                        return
                    evicting.synchronize()
            queue.pop(0)
            self._release_pending.discard(req.kv.req_pool_idx)
            with device_module.stream(self._release_stream):
                self.release_request(req, fenced=True)
                release()
            # Whatever the caller's stream does next with the freed slots follows.
            device_module.current_stream().wait_stream(self._release_stream)

    def release_request(self, req: Req, fenced: bool = False):
        """Free what a request holds here, whether finished, aborted or retracted."""
        if req.hisparse_staging:
            self._release_staging(req)
            return

        # release resources only after the execution of a potential overlapped batch
        # (fenced: the caller saw that batch and the pending backup complete)
        if not fenced:
            if self.decode_producer_stream is not None:
                device_module.current_stream().wait_stream(self.decode_producer_stream)
            self.wait_for_pending_backup()
        if self._resident and self._resident.pop(req.kv.req_pool_idx, None) is not None:
            # its device slots stay mapped; release_kv_cache frees them as it does a prefill's.
            self.req_resident[req.kv.req_pool_idx] = 0
            if self.enable_resident:
                self.paging.release(req.kv.req_pool_idx)
            self._hs_dev_len.pop(req.kv.req_pool_idx, None)
            if self.hs_exclusive:
                # Host slots sit wherever a page left the device pool, not in a prefix.
                self.req_to_host_pool_allocated_len[req.kv.req_pool_idx] = (
                    req.kv.kv_allocated_len
                )

        # Use kv_allocated_len (not seqlen): under speculative decoding the
        # allocator can over-allocate beyond the committed seqlen, and those
        # extra slots may carry stale mapping entries pointing at buffer slots
        # we just freed via free_hisparse_indices(all_hi). If left set, the
        # subsequent release_kv_cache -> allocator.free -> free_hisparse path
        # re-frees them (double-free into the page allocator's free list).
        allocated_len = req.kv.kv_allocated_len
        req_pool_idx = req.kv.req_pool_idx
        allocated_locs = self.req_to_token_pool.req_to_token[
            req_pool_idx, :allocated_len
        ]
        # A PD transfer that ended before its buffer was built.
        self._give_back_buffer_hold(req_pool_idx)

        # The spec commit kernel rotates physical locations between each
        # layer's persistent cache and scratch.  Their current union is the
        # request's allocation ownership; the original side-buffer row is no
        # longer authoritative after the first rotation.
        current_cap = int(self.req_device_buffer_size[req_pool_idx])
        if current_cap > 0:
            owned_locs = self.req_device_buffer_token_locs[
                :, req_pool_idx, :current_cap
            ].reshape(-1)
            owned_locs = self.spec_swap.extend_owned_locs(req_pool_idx, owned_locs)
            all_hi = torch.unique(owned_locs[owned_locs > 0])
            if all_hi.numel() > 0:
                self.token_to_kv_pool_allocator.free_hisparse_indices(all_hi)

            compressed_locs = (
                self.mem_pool_device.translate_loc_from_full_to_compressed(
                    allocated_locs
                )
            )
            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = 0
        else:
            # The request can finish or be aborted during prefill, before a
            # device side buffer exists.  Its ordinary prefill mapping is still
            # owned by release_kv_cache; only release our separate workspace.
            self.spec_swap.free_unrotated_scratch(req_pool_idx)

        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.kv.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.kv.req_pool_idx],
        )
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)

        # clear req info
        self.req_device_buffer_tokens[:, req.kv.req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req.kv.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.kv.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.kv.req_pool_idx] = 0
        self.req_to_host_pool[req.kv.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.kv.req_pool_idx] = 0
        self.lru_slots[:, req.kv.req_pool_idx, :].copy_(self._lru_init)
        self.spec_swap.clear_scratch(req_pool_idx)
        self._skip_first_backup[req.kv.req_pool_idx] = False

    def _run_swap_in_kernel(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        record_plan: bool = False,
    ) -> torch.Tensor:
        """Run the full plan+IO swap-in kernel for one layer; return its slot table.

        record_plan (set on the anchor of a shared-index group) also records the
        miss plan into self._miss_{src,dst,count} for the skip layers to replay.
        """
        num_reqs = req_pool_indices.size(0)
        plan = (
            dict(
                miss_src=self._miss_src[:num_reqs],
                miss_dst=self._miss_dst[:num_reqs],
                miss_count=self._miss_count[:num_reqs],
            )
            if record_plan
            else {}
        )

        seq_lens = compressed_seq_lens
        if self.enable_resident:
            # A resident request has no hot buffer: the swap-in kernels get its
            # queries with length zero and its rows are filled in afterwards.
            compressed_seq_lens = mask_resident_seq_lens(
                seq_lens=seq_lens,
                req_pool_indices=req_pool_indices,
                resident=self.req_resident,
            )
        if top_k_result.ndim == 3:
            top_k_indices = self.spec_swap.swap_in(
                req_pool_indices,
                compressed_seq_lens,
                top_k_result,
                layer_id,
                **plan,
            )
            self._fill_resident_rows(
                req_pool_indices, seq_lens, top_k_result, top_k_indices
            )
            return top_k_indices
        if top_k_result.ndim != 2:
            raise ValueError("HiSparse top-k must be two- or three-dimensional.")

        top_k_indices = self.top_k_device_locs_buffer[:num_reqs, : self.top_k]
        swap_in_fn = (
            load_cache_to_device_buffer_dsv4_mla
            if self.is_dsv4_hisparse
            else load_cache_to_device_buffer_mla
        )
        skip_io_kwargs = {} if _is_xpu else dict(skip_io=self.skip_io)
        swap_in_fn(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
            host_cache=self.mem_pool_host.kv_buffer[layer_id],
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            lru_slots=self.lru_slots[layer_id],
            item_size_bytes=self.item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            page_size=1,
            block_size=self.swap_in_block_size,
            num_real_reqs=self.num_real_reqs,
            **skip_io_kwargs,
            **plan,
        )
        self._fill_resident_rows(
            req_pool_indices, seq_lens, top_k_result, top_k_indices
        )
        return top_k_indices

    def _fill_resident_rows(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> None:
        """Give the resident requests of a swap-in table the rows that hold their
        selected positions: where the allocator's map places them."""
        if not self.enable_resident:
            return
        translate_resident_rows(
            top_k=top_k_result,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            resident=self.req_resident,
            req_to_token=self.req_to_token_pool.req_to_token,
            mapping=self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping,
            num_real_reqs=self.num_real_reqs,
            output=top_k_indices,
        )

    def swap_in_selected_blocks(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_blocks: torch.Tensor,
        layer_id: int,
        sparse_block_size: int,
    ) -> torch.Tensor:
        assert self.is_m3_hisparse
        num_reqs = req_pool_indices.size(0)
        num_selected_tokens = top_k_blocks.size(1) * sparse_block_size
        assert num_selected_tokens <= self.device_buffer_size, (
            f"MiniMax M3 selected {num_selected_tokens} tokens, but the "
            f"HiSparse device buffer holds only {self.device_buffer_size}."
        )
        top_k_indices = self.top_k_device_locs_buffer[:num_reqs, :num_selected_tokens]
        host_layer = layer_id - self.mem_pool_device.start_layer
        load_blocks_to_device_buffer_mha(
            top_k_blocks=top_k_blocks,
            device_buffer_tokens=self.req_device_buffer_tokens[host_layer],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[host_layer],
            host_cache_k=self.mem_pool_host.k_buffer[host_layer],
            host_cache_v=self.mem_pool_host.v_buffer[host_layer],
            device_buffer_k=self.mem_pool_device.layer_router.read(layer_id).key(),
            device_buffer_v=self.mem_pool_device.layer_router.read(layer_id).value(),
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            lru_slots=self.lru_slots[host_layer],
            item_size_bytes=self.item_size_bytes,
            hot_buffer_size=self.device_buffer_size,
            sparse_block_size=sparse_block_size,
            page_size=1,
            block_size=self.swap_in_block_size,
            num_real_reqs=self.num_real_reqs,
            skip_io=self.skip_io,
        )
        return top_k_indices

    def _run_copy_only_kernel(self, num_reqs: int, skip_layer: int) -> None:
        """Replay the anchor's recorded miss plan into a skip layer's buffers
        (IO-only; the anchor's slot table stays valid -- lockstep layout)."""
        copy_cache_planned_mla(
            miss_src=self._miss_src[:num_reqs],
            miss_dst=self._miss_dst[:num_reqs],
            miss_count=self._miss_count[:num_reqs],
            num_real_reqs=self.num_real_reqs,
            host_cache=self.mem_pool_host.kv_buffer[skip_layer],
            device_buffer=self.mem_pool_device.kv_buffer[skip_layer],
            item_size_bytes=self.item_size_bytes,
            num_blocks=self._prefetch_copy_blocks,
            is_dsv4_layout=self.is_dsv4_hisparse,
            skip_io=self.skip_io,
        )

    def swap_in_selected_pages(
        self, req_pool_indices, compressed_seq_lens, top_k_result, layer_id
    ):
        """the slots of _hs_swap_in_selected_pages, through the layer group's redirects."""
        locs = self._hs_swap_in_selected_pages(
            req_pool_indices, compressed_seq_lens, top_k_result, layer_id
        )
        if self.enable_resident:
            locs = self.paging.redirect(locs, layer_id)
            locs = self.paging.read_through_row_cache(locs, layer_id)
        return locs

    def _hs_swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Swap selected top-k tokens into device memory and return their indices.

        With prefetch enabled, anchors swap in synchronously (recording the miss
        plan) and prefetch their skip layers' copies; skip layers just wait.
        """
        if self._pool_layer_index is not None:
            layer_id = self._pool_layer_index[layer_id]
        if (
            self.enable_resident
            and top_k_result.ndim in (2, 3)
            and get_capture_attention_variant() == HISPARSE_RESIDENT
        ):
            # The all-resident graph's top-k already emitted the rows to read.
            return top_k_result
        if not self.enable_prefetch:
            return self._run_swap_in_kernel(
                req_pool_indices,
                compressed_seq_lens,
                top_k_result,
                layer_id,
            )

        num_reqs = req_pool_indices.size(0)
        if self._is_shared_index_layer[layer_id]:
            # Skip layer: wait for its prefetched copy; the anchor's slot table
            # applies (shared index + lockstep buffers).
            slot = self._prefetch_slot[layer_id]
            self._prefetch_events[slot].wait(device_module.current_stream())
            if top_k_result.ndim == 3:
                return self.spec_swap.output_locs(top_k_result, num_reqs)
            return self.top_k_device_locs_buffer[:num_reqs, : self.top_k]

        # Anchor: swap in synchronously (recording the plan), then prefetch the
        # skip layers' copies on the side stream.
        group = self._prefetch_groups.get(layer_id)
        anchor_locs = self._run_swap_in_kernel(
            req_pool_indices,
            compressed_seq_lens,
            top_k_result,
            layer_id,
            record_plan=group is not None,
        )
        if group:
            # Fork: the prefetch stream must observe the anchor's plan (produced
            # on the current stream) before replaying it.
            self.prefetch_stream.wait_stream(device_module.current_stream())
            with device_module.stream(self.prefetch_stream):
                for skip_layer in group:
                    self._run_copy_only_kernel(num_reqs, skip_layer)
                    self._prefetch_events[self._prefetch_slot[skip_layer]].record(
                        self.prefetch_stream
                    )
        return anchor_locs
