from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    HiSparseC4DevicePool,
)
from sglang.srt.mem_cache.hisparse_shard import HiSparseShardView
from sglang.srt.mem_cache.hisparse_memory_pool import (
    HiSparseDSATokenToKVPool,
    HiSparseMHAMainPool,
)
from sglang.srt.mem_cache.stream_handoff import handed_over
from sglang.srt.mem_cache.utils import get_num_new_pages
from sglang.srt.platforms import current_platform

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool


class HiSparseTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    _reclaimer = None  # (class default for allocators built without __init__)
    _shard = HiSparseShardView()  # (class default) the context is not split over GPUs
    _ledger = None  # (class default) split context: the admission numbers every GPU agrees on
    _buffered_prompt_tokens = 0  # (class default)

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        kvcache: HiSparseDSATokenToKVPool | MiniMaxSparseKVPool,
        need_sort: bool,
        host_to_device_ratio: int = 2,
        shard: HiSparseShardView = HiSparseShardView(),
    ):
        self._kvcache = kvcache
        # Split context: `size`, the map and the device allocator count this GPU's rows;
        # the logical allocator hands out widened slots, a page of them per page of rows.
        self._shard = shard
        # a fractional ratio gives a page-aligned integer logical size (the index-K buffer uses the same rule)
        self._size_full = int(size * host_to_device_ratio) // page_size * page_size
        self._size_hisparse = size
        self.compress_ratio = 1
        self.dtype = dtype
        self.device = device
        self.page_size = shard.tokens(page_size)
        self.need_sort = need_sort
        self._spec_scratch_capacity = 0
        self._spec_device_buffer_size = 0
        self._reclaimer = None  # the coordinator, which can move resident pages to host
        # Exclusive placement: prompts up to this length stay buffered and hold host
        # slots besides their device slots (0: every token has a host slot anyway).
        self._buffered_prompt_tokens = 0

        paged_allocator_cls = (
            current_platform.get_paged_allocator_cls() or PagedTokenToKVPoolAllocator
        )
        self.logical_attn_allocator = paged_allocator_cls(
            shard.tokens(self._size_full),
            self.page_size,
            self.dtype,
            self.device,
            kvcache,
            need_sort,
        )
        # the device allocator spans the band too; only mapped pages are handed out.
        # A hybrid-linear model keeps its DSA layers in an inner pool: the ranges are there.
        self._hs_el = getattr(self.get_hisparse_kvcache(), "_hs_el", None)
        self.hisparse_attn_allocator = paged_allocator_cls(
            self._hs_el["k_max"] if self._hs_el else self._size_hisparse,
            self.row_page_size,
            self.dtype,
            self.device,
            kvcache,
            need_sort,
        )
        if shard.split and (
            self._hs_el is None or self.get_hisparse_kvcache().hs_shard != shard
        ):
            raise ValueError(
                "HiSparse with a split context needs the paged pool built for the same shard."
            )
        if self._hs_el is not None:
            self._hs_el_init_limits()
        self.full_to_hisparse_device_index_mapping = torch.cat(
            [
                torch.zeros(
                    self._size_full + self.row_page_size,
                    dtype=torch.int64,
                    device=self.device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=self.device),
            ]
        )

        self.free_pages = None
        self.release_pages = None
        self.free_group = None
        self.clear()
        self._kvcache.register_mapping(
            weakref.proxy(self.full_to_hisparse_device_index_mapping)
        )

    def configure_spec_scratch(
        self, capacity: int, *, device_buffer_size: int = 0
    ) -> None:
        if any(
            size < 0 or size % self.row_page_size for size in (capacity, device_buffer_size)
        ):
            raise ValueError(
                "HiSparse spec scratch and device buffer sizes must be "
                f"non-negative multiples of page_size={self.row_page_size}."
            )
        self._spec_scratch_capacity = capacity
        self._spec_device_buffer_size = device_buffer_size if capacity else 0

    def request_slot_reserve(
        self, *, has_req_pool_slot: bool, prefill_tokens: int = 0
    ) -> int:
        paged_prefill = -(-prefill_tokens // self.row_page_size) * self.row_page_size
        # Prefill KV is reused by the persistent buffer. Charge only its
        # missing slots, including on chunks whose scratch was already allocated.
        buffer_reserve = max(self._spec_device_buffer_size - paged_prefill, 0)
        scratch_reserve = 0 if has_req_pool_slot else self._spec_scratch_capacity
        return (
            buffer_reserve
            + scratch_reserve
            + self._buffered_host_reserve(
                has_req_pool_slot=has_req_pool_slot, prefill_tokens=prefill_tokens
            )
        )

    def _buffered_host_reserve(
        self, *, has_req_pool_slot: bool, prefill_tokens: int
    ) -> int:
        """Exclusive placement: a prompt that stays buffered takes a host slot per
        token on top of the device slots its prefill is charged for, and waits
        while the host side cannot take it. It holds them once it has its slot."""
        if has_req_pool_slot or not 0 < prefill_tokens <= self._buffered_prompt_tokens:
            return 0
        reserve = -(-prefill_tokens // self.row_page_size) * self.row_page_size
        if self._reclaimer is not None:
            reserve += self._reclaimer.exclusive_host_shortfall(
                device_free=self._exclusive_device_free()
            )
        return reserve

    def check_decode_capacity(
        self, *, num_tokens: int, tree_cache, requests=None, spec_algorithm=None
    ) -> bool:
        if self._decode_step_fits(num_tokens, tree_cache, requests, spec_algorithm):
            return True
        if self._reclaimer is None:
            return False
        # Ended requests may still hold their pages: take them before a retraction.
        self._reclaimer.run_deferred_releases(block=True)
        return self._decode_step_fits(num_tokens, tree_cache, requests, spec_algorithm)

    def _decode_step_fits(
        self, num_tokens: int, tree_cache, requests, spec_algorithm
    ) -> bool:
        fits = super().check_decode_capacity(
            num_tokens=num_tokens,
            tree_cache=tree_cache,
            requests=requests,
            spec_algorithm=spec_algorithm,
        )
        # Exclusive placement: the buffered requests' host pages are not in that count.
        if fits and self._buffered_prompt_tokens and requests is not None:
            fits = self._reclaimer.exclusive_host_fits(
                requests, device_free=self._exclusive_device_free()
            )
        return fits

    @property
    def size_full(self) -> int:
        return self._shard.tokens(self._size_full)

    @property
    def size(self) -> int:
        return self._shard.tokens(self._size_full)

    @property
    def shard(self) -> HiSparseShardView:
        return self._shard

    @property
    def row_page_size(self) -> int:
        """Rows of this GPU per page; page_size counts tokens (the same without a split)."""
        return self.page_size // self._shard.count

    @property
    def rows_full(self) -> int:
        """Logical rows of this GPU: what the map and the paging tables are sized by."""
        return self._size_full

    def set_reclaimer(self, reclaimer) -> None:
        self._reclaimer = reclaimer

    def set_ledger(self, ledger) -> None:
        self._ledger = ledger


    def device_free_slots(self) -> int:
        return self._device_free()

    def _device_free(self) -> int:
        """Free device slots, with the scratch blocks kept for the next arrivals:
        `reclaim_device` gives those back as soon as slots are short."""
        spare = self._reclaimer.spare_scratch_slots() if self._reclaimer else 0
        return self.hisparse_attn_allocator.available_size() + spare

    def available_size(self) -> int:
        if self._ledger is not None:
            return self._ledger.available_tokens()
        if self._reclaimer is not None and self._reclaimer.hs_exclusive is True:
            return self._exclusive_available_size()
        # Device slots that moving resident pages to host would free count as available for admission.
        reclaimable = (
            self._reclaimer.reclaimable_slots() if self._reclaimer is not None else 0
        )
        if self._hs_el is not None:  # admission sees the static layout
            el = self._hs_el
            logical = (
                self.logical_attn_allocator.available_size()
                + max(0, self.logical_attn_allocator.num_pages - el["log_max_page"])
                * self.page_size
            )
            band = self._hs_el_mapped_rows() - (el["k_min"] + self.row_page_size)
            return min(logical, self._device_free() + reclaimable - max(0, band))
        return min(
            self.logical_attn_allocator.available_size(),
            self._device_free() + reclaimable,
        )

    def create_prefill_budget(self, tree_cache, *, num_mixed_decode_tokens=0):
        if self._reclaimer is None:
            return super().create_prefill_budget(
                tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
            )
        from sglang.srt.mem_cache.prefill_budget import HiSparsePagedPrefillBudget

        return HiSparsePagedPrefillBudget(
            self, tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
        )

    def holds_request_tokens(self) -> bool:
        """Whether a request holds logical slots, which it gives back when it ends."""
        if self._ledger is not None:
            return self._ledger.holds_request_tokens()
        logical = self.logical_attn_allocator
        free = logical.available_size()
        if self._hs_el is not None:  # logical pages past the mapped index keys are free
            free += (
                max(0, logical.num_pages - self._hs_el["log_max_page"]) * self.page_size
            )
        return free < self.size_full

    def _exclusive_device_free(self) -> int:
        """Free device slots of the static layout (the band is lent memory)."""
        el = self._hs_el
        band = self._hs_el_mapped_rows() - (el["k_min"] + self.row_page_size)
        return self._device_free() - max(0, band)

    def _exclusive_available_size(self) -> int:
        """Exclusive placement: a token needs a logical slot and a device slot or a
        host slot; the coordinator prices what the two pools can still take."""
        el = self._hs_el
        logical = (
            self.logical_attn_allocator.available_size()
            + max(0, self.logical_attn_allocator.num_pages - el["log_max_page"])
            * self.page_size
        )
        # The coordinator prices rows of this GPU; the scheduler asks in tokens.
        shard = self._shard
        return shard.tokens(
            self._reclaimer.exclusive_available_slots(
                logical_free=shard.whole(logical),
                device_free=self._exclusive_device_free(),
            )
        )

    # ---- elastic range ------------------------------------------------------------------------------------------------
    @staticmethod
    def _hs_el_filter(alloc, max_page: int) -> None:
        alloc.merge_and_sort_free()
        alloc.free_pages = alloc.free_pages[alloc.free_pages <= max_page]

    def _hs_el_mapped_rows(self) -> int:
        """Device rows the allocator may hand out: the mapped rows, but not the
        whole pages the last mapped chunk can hold past the band's last page
        (the chunk granularity does not divide a page of rows); the paging
        tables hold one entry per frame up to that page."""
        el = self._hs_el
        return min(el["kv"].mapped // el["row"], el["k_max"] + self.row_page_size)

    def _hs_el_init_limits(self) -> None:
        el = self._hs_el
        P = self.row_page_size
        el["dev_max_page"] = self._hs_el_mapped_rows() // P - 1
        el["log_max_page"] = min(
            self.logical_attn_allocator.num_pages,
            el["ik"].mapped // el["ik_page_bytes"] - 1,
        )
        self._hs_el_filter(self.hisparse_attn_allocator, el["dev_max_page"])
        self._hs_el_filter(self.logical_attn_allocator, el["log_max_page"])
        for alloc, key in (
            (self.hisparse_attn_allocator, "dev_max_page"),
            (self.logical_attn_allocator, "log_max_page"),
        ):
            orig = alloc.merge_and_sort_free

            def merge(
                orig=orig, alloc=alloc, key=key
            ):  # pages above the mapped limit never come back
                if not el["pinned"]:
                    orig()
                    alloc.free_pages = alloc.free_pages[alloc.free_pages <= el[key]]
                    return
                # Pinned ranges: the limit never moves and no freed page lies above
                # it; the mask's read-back would wait for the batch in flight.
                orig()

            alloc.merge_and_sort_free = merge
            # Every freed page passes through _release_page_ids (free, free_page_ids,
            # free groups; both need_sort modes). A page above the current limit is
            # unmapped or about to be (an eviction inside a band take-back frees pages
            # above the lowered limit); it must not re-enter the free list, where
            # need_sort=False puts it first in line. Raising the limit adds the range back.
            orig_rel = alloc._release_page_ids

            def release(*page_ids, orig_rel=orig_rel, key=key):
                if el["pinned"]:
                    orig_rel(*page_ids)
                else:
                    orig_rel(*[p[p <= el[key]] for p in page_ids])

            alloc._release_page_ids = release
            if key == "dev_max_page":
                el["dev_release_unfiltered"] = orig_rel

    def hs_el_set_limit(self, which: str, new_max: int) -> None:
        """Raise (add the new pages) or lower (drop pages above; the caller made them free) an allocator's limit."""
        el = self._hs_el
        alloc = (
            self.hisparse_attn_allocator
            if which == "dev"
            else self.logical_attn_allocator
        )
        key = "dev_max_page" if which == "dev" else "log_max_page"
        if which == "dev":
            new_max = min(new_max, el["k_max"] // self.row_page_size)
        old = el[key]
        if new_max > old:
            alloc.merge_and_sort_free()  # under the old limit: freed pages above it are dropped, not duplicated
            alloc.free_pages = alloc.free_pages[
                alloc.free_pages <= old
            ]  # never two copies of an added page
            el[key] = new_max
            add = torch.arange(
                old + 1, new_max + 1, dtype=torch.int64, device=self.device
            )
            alloc.free_pages, _ = torch.sort(torch.cat((alloc.free_pages, add)))
        elif new_max < old:
            el[key] = new_max
            self._hs_el_filter(alloc, new_max)

    def hs_alloc_below(self, need_size: int, max_slot: int):
        """need_size slots from pages entirely below max_slot (None if there are not enough)."""
        alloc = self.hisparse_attn_allocator
        P = self.row_page_size
        n = need_size // P
        alloc.merge_and_sort_free()
        el = self._hs_el
        if el is not None and el["pinned"] and max_slot // P > el["dev_max_page"]:
            # No band: every free page lies below; the mask would read back,
            # which waits for the batch in flight.
            if len(alloc.free_pages) < n:
                return None
            take = alloc.free_pages[:n]
            alloc.free_pages = alloc.free_pages[n:]
            return (take[:, None] * P + torch.arange(P, device=self.device)).reshape(-1)
        ok = alloc.free_pages < (max_slot // P)
        cand = alloc.free_pages[ok]
        if len(cand) < n:
            return None
        take = cand[:n]
        alloc.free_pages = torch.cat((cand[n:], alloc.free_pages[~ok]))
        alloc.free_pages, _ = torch.sort(alloc.free_pages)
        return (take[:, None] * P + torch.arange(P, device=self.device)).reshape(-1)

    def hs_alloc_above(self, need_size: int, min_slot: int):
        """need_size slots from pages entirely at or above min_slot (None if there are not enough)."""
        alloc = self.hisparse_attn_allocator
        P = self.row_page_size
        n = need_size // P
        alloc.merge_and_sort_free()
        ok = alloc.free_pages >= -(-min_slot // P)
        cand = alloc.free_pages[ok]
        if len(cand) < n:
            return None
        take = cand[:n]
        alloc.free_pages, _ = torch.sort(torch.cat((alloc.free_pages[~ok], cand[n:])))
        return (take[:, None] * P + torch.arange(P, device=self.device)).reshape(-1)

    def hs_copy_rows(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """copy device rows src -> dst in every layer."""
        for buf in self.get_hisparse_kvcache().kv_buffer:
            buf[dst] = buf[src]

    def get_kvcache(self):
        return self._kvcache

    def get_hisparse_kvcache(self) -> HiSparseDSATokenToKVPool | HiSparseMHAMainPool:
        from sglang.srt.mem_cache.memory_pool import (
            HybridLinearKVPool,
            MiniMaxSparseKVPool,
        )

        if isinstance(self._kvcache, MiniMaxSparseKVPool):
            return self._kvcache.main_pool
        # A hybrid linear model keeps its DSA layers in a dense full pool.
        if isinstance(self._kvcache, HybridLinearKVPool):
            return self._kvcache.full_kv_pool
        return self._kvcache

    def alloc(self, need_size: int):
        if self.row_page_size != 1:
            raise NotImplementedError(
                "HiSparse generic allocation is only supported for page_size=1. "
                "Use alloc_extend for paged allocation."
            )

        logical_indices = self.logical_attn_allocator.alloc(need_size)
        if logical_indices is None:
            return None

        hisparse_indices = self.hisparse_attn_allocator.alloc(need_size)
        if hisparse_indices is None:
            self.logical_attn_allocator.free(logical_indices)
            return None

        self.full_to_hisparse_device_index_mapping[logical_indices] = hisparse_indices
        return logical_indices

    def alloc_logical_only(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        """Allocate only logical indices without hisparse device indices."""
        if self._hs_el is not None and self._reclaimer is not None:
            self._reclaimer.hs_el_ensure_logical(
                get_num_new_pages(
                    seq_lens=seq_lens_cpu,
                    page_size=self.page_size,
                    prefix_lens=prefix_lens_cpu,
                )
            )
        return self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )

    def alloc_device_buffer(self, allocated_indices, need_size: int):
        assert need_size % self.row_page_size == 0
        # clear original reference and isolate the buffer from outside addressing, allocate new buffer if needed
        hisparse_indices = self.full_to_hisparse_device_index_mapping[allocated_indices]
        self.full_to_hisparse_device_index_mapping[allocated_indices] = 0
        # Zero means unmapped; after alloc_logical_only the mapping is all zeros.
        hisparse_indices = hisparse_indices[hisparse_indices > 0]
        if len(hisparse_indices) >= need_size:
            buffer_indices = hisparse_indices[:need_size]
            self.free_hisparse_indices(hisparse_indices[need_size:])
            if self._hs_el is not None:
                buffer_indices = self._hs_el_buffer_below(buffer_indices)
        else:
            # page alignment, claiming the residual space for an incomplete page
            page_residual_length = len(hisparse_indices) % self.row_page_size
            if page_residual_length != 0:
                hisparse_indices = torch.cat(
                    [
                        hisparse_indices,
                        torch.arange(
                            hisparse_indices[-1] + 1,
                            hisparse_indices[-1]
                            + self.row_page_size
                            - page_residual_length
                            + 1,
                            device=self.device,
                        ),
                    ]
                )
            need_extra = need_size - len(hisparse_indices)
            if (  # admission counted resident requests' pages as free; move pages to host to free them
                self._reclaimer is not None
                and getattr(self._reclaimer, "hs_spec_res", False) is True
                and self.hisparse_attn_allocator.available_size() < need_extra
            ):
                self._reclaimer.hs_spec_reclaims[1] += 1
                self._reclaimer.reclaim_device(need_extra)
            if self._hs_el is not None:  # buffers live below the band
                if len(hisparse_indices):
                    hisparse_indices = self._hs_el_buffer_below(hisparse_indices)
                extra_indices = self._hs_el_alloc_below_or_evict(need_extra)
            else:
                extra_indices = self.hisparse_attn_allocator.alloc(need_extra)
            assert extra_indices is not None, (
                "Hisparse allocation failed in alloc_device_buffer"
            )
            buffer_indices = torch.cat([hisparse_indices, extra_indices])
        return buffer_indices

    def alloc_buffer_extra(self, prefill_tokens: int):
        """The slots `alloc_device_buffer` adds to this prompt's prefill pages
        (`request_slot_reserve`'s buffer charge), taken now: empty when the buffer
        needs none, None when they do not fit."""
        paged_prefill = -(-prefill_tokens // self.row_page_size) * self.row_page_size
        need = max(self._spec_device_buffer_size - paged_prefill, 0)
        if need == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        if (
            self._reclaimer is not None
            and self.hisparse_attn_allocator.available_size() < need
        ):
            self._reclaimer.reclaim_device(need)
        if self._hs_el is not None:  # buffers live below the band
            return self._hs_el_alloc_below_or_evict(need)
        return self.hisparse_attn_allocator.alloc(need)

    def _hs_el_alloc_below_or_evict(self, need_size: int):
        floor = self._hs_el["k_min"] + self.row_page_size
        out = self.hs_alloc_below(need_size, floor)
        while (
            out is None
            and self._reclaimer is not None
            and self._reclaimer.hs_el_evict_below_band()
        ):
            out = self.hs_alloc_below(need_size, floor)
        return out

    def _hs_el_buffer_below(self, slots: torch.Tensor) -> torch.Tensor:
        """a buffer must not sit in the band (it is not reachable through the logical map, so the band
        could not be taken back): pages at or above the static pool are copied to fresh pages below it."""
        if self._hs_el["pinned"]:  # no band to sit in
            return slots
        floor = self._hs_el["k_min"] + self.row_page_size
        P = self.row_page_size
        high = slots >= floor
        if not bool(high.any()):
            return slots
        hs = slots[high]
        pages = torch.unique(hs // P)
        new = self._hs_el_alloc_below_or_evict(len(pages) * P)
        if new is None:
            raise RuntimeError("HiSparse elastic: no room below the band for a buffer")
        new_pages = new.view(-1, P)[:, 0] // P
        src = (pages[:, None] * P + torch.arange(P, device=self.device)).reshape(-1)
        dst = (new_pages[:, None] * P + torch.arange(P, device=self.device)).reshape(-1)
        self.hs_copy_rows(src, dst)
        self.free_hisparse_indices(src)
        idx = torch.searchsorted(pages, hs // P)
        out = slots.clone()
        out[high] = new_pages[idx] * P + hs % P
        if self._reclaimer is not None:
            self._reclaimer.hs_el_stats[3] += len(pages)
        return out

    def free_buffer_extra(self, slots: torch.Tensor) -> None:
        """Give back slots `alloc_buffer_extra` handed out: whole pages below the
        band, so neither the padding page nor a page above the limit is among
        them and no filter has to read them back."""
        if slots.numel() == 0:
            return
        alloc = self.hisparse_attn_allocator
        # Taken at pre-allocation on the schedule stream, given back by a release on its own.
        pages = handed_over(slots)[:: self.row_page_size] // self.row_page_size
        if self._hs_el is None or alloc.free_group is not None:
            alloc.free_page_ids(pages)
        else:
            self._hs_el["dev_release_unfiltered"](pages)

    def free_hisparse_indices(self, buffer_indices: torch.Tensor):
        # Page zero is the padding sink and is never owned by a request.
        self.hisparse_attn_allocator.free(
            buffer_indices[buffer_indices >= self.row_page_size]
        )

    def free_device_pages(self, page_ids: torch.Tensor) -> None:
        """Whole device pages, each once and none the padding sink: no mask and
        no dedup, so nothing here reads back. In ascending order, as
        free_hisparse_indices returns them: the order frames come back in
        decides which frame a later page lands in."""
        self.hisparse_attn_allocator.free_page_ids(torch.sort(page_ids).values)

    def get_last_loc_compressed(self, last_locs: torch.Tensor):
        return last_locs

    def get_last_loc_hisparse_device(self, last_locs: torch.Tensor):
        return self._kvcache._translate_loc_to_hisparse_device(last_locs)

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
        extend_num_tokens: int,
    ):
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            prefix_lens=prefix_lens_cpu,
        )
        if self._hs_el is not None and self._reclaimer is not None:
            self._reclaimer.hs_el_ensure_logical(num_new_pages)
        if (
            num_new_pages
            > self.logical_attn_allocator.available_size() // self.page_size
        ):
            return None
        if (
            self._hs_el is not None
            and self._reclaimer is not None
            and num_new_pages
            > self.hisparse_attn_allocator.available_size() // self.row_page_size
        ):  # keep the index-K this allocation's logical pages need (ensure_logical just mapped it)
            self._reclaimer.hs_el_grow_band(keep_extra=num_new_pages)
        if (
            self._reclaimer is not None  # move resident pages to host to make room
            and num_new_pages
            > self.hisparse_attn_allocator.available_size() // self.row_page_size
        ):
            self._reclaimer.reclaim_device(num_new_pages * self.row_page_size)
        if (
            num_new_pages
            > self.hisparse_attn_allocator.available_size() // self.row_page_size
        ):
            if self._ledger is not None:
                # A None here would be this GPU's answer alone: the others allocated.
                raise RuntimeError(
                    "HiSparse paged (split context): no device frames for "
                    f"{num_new_pages} pages the ledger admitted"
                )
            return None

        logical_indices = self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        assert logical_indices is not None, "Logical allocation failed in alloc_extend"

        # Device rows mirror the logical rows the chunk opens (no split: every slot).
        shard = self._shard
        logical_rows = shard.anchor_rows(logical_indices)
        hisparse_last_loc = self.get_last_loc_hisparse_device(shard.row_of(last_loc))
        hisparse_indices = self.hisparse_attn_allocator.alloc_extend(
            shard.extent(prefix_lens),
            shard.extent(prefix_lens_cpu),
            shard.extent(seq_lens),
            shard.extent(seq_lens_cpu),
            hisparse_last_loc,
            len(logical_rows),
            num_new_pages=num_new_pages,
        )
        assert hisparse_indices is not None, (
            "Hisparse allocation failed in alloc_extend"
        )
        self.full_to_hisparse_device_index_mapping[logical_rows] = hisparse_indices
        return logical_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
    ):
        if self._hs_el is not None and self._reclaimer is not None:
            self._reclaimer.hs_el_ensure_logical(len(seq_lens_cpu) + 1)
        return self.logical_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )

    host_row0 = None  # map entries at or above this row index name host rows

    def free_hisparse(self, free_indices: torch.Tensor):
        # Split context: a row goes when its anchor slot does (no split: every slot is a row).
        free_rows = self._shard.anchor_rows(free_indices)
        hisparse_indices = self._kvcache._translate_loc_to_hisparse_device(free_rows)
        hisparse_indices = hisparse_indices[hisparse_indices > 0]
        if self.host_row0 is not None:  # pages on host hold no device slot
            hisparse_indices = hisparse_indices[hisparse_indices < self.host_row0]
        self.free_hisparse_indices(hisparse_indices)
        self.full_to_hisparse_device_index_mapping[free_rows] = 0

    def hs_keep_reserved(self, slots: torch.Tensor) -> None:
        """Device slots taken for the server's lifetime (row-cache scratch rows): clear() must not hand them out."""
        pages = torch.unique(slots.to(torch.int64) // self.row_page_size)
        prev = getattr(self, "_hs_reserved_pages", None)
        self._hs_reserved_pages = (
            pages if prev is None else torch.unique(torch.cat((prev, pages)))
        )

    def reuse_fences(self) -> list:
        if self._reclaimer is None:
            return []
        return self._reclaimer.frame_reuse_fences()

    def release_ended_requests(self) -> None:
        if self._reclaimer is not None:
            self._reclaimer.run_deferred_releases(block=True)

    def clear(self):
        self.release_ended_requests()
        if self._reclaimer is not None:  # their slots are in the rebuilt free list
            self._reclaimer.forget_spare_scratch()
        self.logical_attn_allocator.clear()
        self.hisparse_attn_allocator.clear()
        res = getattr(self, "_hs_reserved_pages", None)
        if res is not None:  # row-cache scratch rows stay reserved across a cache flush
            dev = self.hisparse_attn_allocator
            dev.free_pages = dev.free_pages[
                ~torch.isin(dev.free_pages, res.to(dev.free_pages.device))
            ]
        if self._hs_el is not None:  # only mapped pages are free
            self._hs_el_filter(
                self.hisparse_attn_allocator, self._hs_el["dev_max_page"]
            )
            self._hs_el_filter(self.logical_attn_allocator, self._hs_el["log_max_page"])
        # Keep the trailing -1: it is what a last_loc of -1 translates to.
        self.full_to_hisparse_device_index_mapping[:-1].fill_(0)
        self.free_group = None

    # No deferred frees: free() must clear the full-to-hisparse mapping at once.
    def free_group_begin(self):
        return

    def free_group_end(self):
        return

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        self.logical_attn_allocator.free(free_index)
        self.free_hisparse(free_index)
        assert (
            self.logical_attn_allocator.available_size()
            <= self.logical_attn_allocator.size
        )
        assert (
            self.hisparse_attn_allocator.available_size()
            <= self.hisparse_attn_allocator.size
        )


class DeepSeekV4HiSparseTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    @property
    def has_window_pool(self) -> bool:
        # Wraps an SWA allocator and exposes its full interface.
        return True

    def __init__(
        self,
        logical_attn_allocator: BaseTokenToKVPoolAllocator,
    ):
        assert isinstance(logical_attn_allocator._kvcache, DeepSeekV4TokenToKVPool)
        assert isinstance(
            logical_attn_allocator._kvcache.c4_kv_pool, HiSparseC4DevicePool
        )
        self.compress_ratio = 4

        self.hisparse_kvcache = logical_attn_allocator._kvcache.c4_kv_pool
        self._size_full = logical_attn_allocator.size_full
        self._size_hisparse = self.hisparse_kvcache.size

        self.dtype = self.hisparse_kvcache.dtype
        self.device = self.hisparse_kvcache.device
        # Keep the public page_size as the logical DSV4 full/SWA page size.
        # C4 HiSparse allocation/device-buffer code must use the compressed page size.
        self.page_size = logical_attn_allocator.page_size
        self.hisparse_page_size = self.hisparse_kvcache.page_size

        self.logical_attn_allocator = logical_attn_allocator
        self._kvcache = logical_attn_allocator._kvcache
        paged_allocator_cls = (
            current_platform.get_paged_allocator_cls() or PagedTokenToKVPoolAllocator
        )
        self.hisparse_attn_allocator = paged_allocator_cls(
            self._size_hisparse,
            self.hisparse_page_size,
            self.dtype,
            self.device,
            self.hisparse_kvcache,
            logical_attn_allocator.need_sort,
        )

        self.full_to_hisparse_device_index_mapping = torch.cat(
            [
                torch.zeros(
                    self._kvcache.c4_logical_size + self.hisparse_page_size,
                    dtype=torch.int64,
                    device=self.device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=self.device),
            ]
        )

        self.need_sort = logical_attn_allocator.need_sort
        self.free_pages = None
        self.release_pages = None
        self.free_group = None
        self.full_free_group = []
        self.clear()

        self.hisparse_kvcache.register_mapping(
            weakref.proxy(self.full_to_hisparse_device_index_mapping)
        )

    @property
    def size_full(self) -> int:
        return self._size_full

    @property
    def size(self) -> int:
        return self.logical_attn_allocator.size

    @property
    def size_swa(self) -> int:
        return self.logical_attn_allocator.size_swa

    @property
    def full_to_swa_index_mapping(self):
        return self.logical_attn_allocator.full_to_swa_index_mapping

    def debug_print(self) -> str:
        msg = self.logical_attn_allocator.debug_print()
        msg += (
            f"#hisparse-available-size: "
            f"{self.hisparse_attn_allocator.available_size()}, "
        )
        return msg

    def get_kvcache(self):
        return self._kvcache

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        return self.logical_attn_allocator.translate_loc_from_full_to_swa(kv_indices)

    def get_hisparse_kvcache(self) -> HiSparseC4DevicePool:
        return self.hisparse_kvcache

    def translate_swa_indices_for_transfer(
        self, kv_indices: torch.Tensor
    ) -> torch.Tensor:
        return self.logical_attn_allocator.translate_swa_indices_for_transfer(
            kv_indices
        )

    def create_prefill_budget(self, tree_cache, *, num_mixed_decode_tokens=0):
        # Use this wrapper's capacity, which also includes the compressed pool.
        from sglang.srt.mem_cache.prefill_budget import SWAPrefillBudget

        return SWAPrefillBudget(
            self, tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
        )

    def swa_capacity_and_available(self, *, full_capacity, swa_capacity):
        return (
            (full_capacity, self.full_available_size()),
            (swa_capacity, self.swa_available_size()),
        )

    def owner_available_size(self):
        """What the token pool can serve now: its full-attention side."""
        return self.full_available_size()

    def full_available_size(self):
        return min(
            self.logical_attn_allocator.full_available_size(),
            self.hisparse_attn_allocator.available_size() * self.compress_ratio,
        )

    def swa_available_size(self):
        return self.logical_attn_allocator.swa_available_size()

    def window_pool_names(self):
        return self.logical_attn_allocator.window_pool_names()

    def reclaim_for_prealloc(self, tree_cache, demand) -> str | None:
        # C4 needs no reclaim here: full_available_size prices it into the budget.
        return self.logical_attn_allocator.reclaim_for_prealloc(tree_cache, demand)

    def free_swa(self, free_indices: torch.Tensor):
        self.logical_attn_allocator.free_swa(free_indices)

    def free_swa_segment(self, free_indices: torch.Tensor, *, start_pos: int):
        self.logical_attn_allocator.free_swa_segment(free_indices, start_pos=start_pos)

    def free_full(self, free_indices: torch.Tensor):
        if free_indices.numel() == 0:
            return

        if self.free_group is None:
            self.logical_attn_allocator.free_full(free_indices)
        else:
            self.full_free_group.append(self._copy_for_free_group(free_indices))

    def available_size(self) -> int:
        return min(
            self.logical_attn_allocator.available_size(),
            self.hisparse_attn_allocator.available_size() * self.compress_ratio,
        )

    def alloc(self, need_size: int):
        raise NotImplementedError(
            "DeepSeek V4 HiSparse allocator does not support direct token allocation; "
            "use alloc_extend or alloc_decode instead."
        )

    def alloc_logical_only(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        """Allocate decode logical indices without allocating C4 hisparse device pages."""
        return self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )

    def alloc_extend_swa_tail(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        swa_tail_len: int,
    ):
        return self.logical_attn_allocator.alloc_extend_swa_tail(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=extend_num_tokens,
            swa_tail_len=swa_tail_len,
        )

    def alloc_device_buffer(self, allocated_indices, need_size: int):
        assert need_size % self.hisparse_page_size == 0
        hisparse_indices = self.full_to_hisparse_device_index_mapping[allocated_indices]
        self.full_to_hisparse_device_index_mapping[allocated_indices] = 0
        hisparse_indices = hisparse_indices[hisparse_indices > 0]

        device_buffer_size = need_size - self.hisparse_page_size
        P = len(hisparse_indices)
        if P > device_buffer_size + 1:
            newest_src = hisparse_indices[P - 1].clone()
            old_at_dbs = hisparse_indices[device_buffer_size].clone()
            hisparse_indices[device_buffer_size] = newest_src
            hisparse_indices[P - 1] = old_at_dbs

        if len(hisparse_indices) >= need_size:
            buffer_indices = hisparse_indices[:need_size]
            surplus = hisparse_indices[need_size:]
            if surplus.numel() > 0:
                buffer_pages = torch.unique(buffer_indices // self.hisparse_page_size)
                surplus_pages = torch.unique(surplus // self.hisparse_page_size)
                pure_surplus = surplus_pages[~torch.isin(surplus_pages, buffer_pages)]
                if pure_surplus.numel() > 0:
                    self.hisparse_attn_allocator.free(
                        pure_surplus * self.hisparse_page_size
                    )
        else:
            page_residual_length = len(hisparse_indices) % self.hisparse_page_size
            if page_residual_length != 0:
                hisparse_indices = torch.cat(
                    [
                        hisparse_indices,
                        torch.arange(
                            hisparse_indices[-1] + 1,
                            hisparse_indices[-1]
                            + self.hisparse_page_size
                            - page_residual_length
                            + 1,
                            device=self.device,
                        ),
                    ]
                )
            extra_indices = self.hisparse_attn_allocator.alloc(
                need_size - len(hisparse_indices)
            )
            assert extra_indices is not None, (
                "Hisparse allocation failed in alloc_device_buffer"
            )
            buffer_indices = torch.cat([hisparse_indices, extra_indices])
        return buffer_indices

    def free_hisparse_indices(self, buffer_indices: torch.Tensor):
        self.hisparse_attn_allocator.free(
            buffer_indices[buffer_indices >= self.hisparse_page_size]
        )

    def get_last_loc_compressed(self, last_locs: torch.Tensor):
        # Last complete C4 block of a prefix of last_loc + 1 tokens; -1 stays -1.
        return (last_locs - (self.compress_ratio - 1)) // self.compress_ratio

    def get_last_loc_hisparse_device(self, last_locs: torch.Tensor):
        return self.hisparse_kvcache._translate_loc_to_hisparse_device(
            self.get_last_loc_compressed(last_locs)
        )

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        assert self.page_size > 1

        num_new_pages_logical = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        num_new_pages_hisparse = get_num_new_pages(
            seq_lens=seq_lens_cpu // self.compress_ratio,
            page_size=self.hisparse_page_size,
            prefix_lens=prefix_lens_cpu // self.compress_ratio,
        )
        if (
            num_new_pages_logical
            > self.logical_attn_allocator.available_size() // self.page_size
        ):
            return None
        if (
            num_new_pages_hisparse
            > self.hisparse_attn_allocator.available_size() // self.hisparse_page_size
        ):
            return None

        logical_indices = self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        assert logical_indices is not None, "Logical allocation failed in alloc_extend"

        compressed_logical_indices = (
            self.hisparse_kvcache.translate_loc_from_full_to_compressed(logical_indices)
        )
        hisparse_last_loc = self.get_last_loc_hisparse_device(last_loc)
        hisparse_indices = self.hisparse_attn_allocator.alloc_extend(
            prefix_lens // self.compress_ratio,
            prefix_lens_cpu // self.compress_ratio,
            seq_lens // self.compress_ratio,
            seq_lens_cpu // self.compress_ratio,
            hisparse_last_loc,
            len(compressed_logical_indices),
        )
        assert hisparse_indices is not None, (
            "Hisparse allocation failed in alloc_extend"
        )

        self.full_to_hisparse_device_index_mapping[compressed_logical_indices] = (
            hisparse_indices.to(torch.int64)
        )
        return logical_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        return self.logical_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )

    def free_compressed(self, compressed_indices: torch.Tensor):
        hisparse_indices = self.hisparse_kvcache.translate_loc_to_hisparse_device(
            compressed_indices
        )
        hisparse_indices = hisparse_indices[hisparse_indices > 0]
        self.free_hisparse_indices(hisparse_indices)
        self.full_to_hisparse_device_index_mapping[compressed_indices] = 0

    def free_hisparse(self, free_indices: torch.Tensor):
        compressed_indices = (
            self.hisparse_kvcache.translate_loc_from_full_to_compressed(free_indices)
        )
        self.free_compressed(compressed_indices)

    def clear(self):
        self.logical_attn_allocator.clear()
        self.hisparse_attn_allocator.clear()

        self.full_to_hisparse_device_index_mapping[:-1].fill_(0)
        self.free_group = None
        self.full_free_group = []

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.free_group is None:
            self.logical_attn_allocator.free(free_index)
            self.free_hisparse(free_index)
        else:
            self.free_group.append(self._copy_for_free_group(free_index))

    def free_group_begin(self):
        super().free_group_begin()
        self.full_free_group = []

    def free_group_end(self):
        super().free_group_end()
        if self.full_free_group:
            full_free_group = self.full_free_group
            self.full_free_group = []
            self.free_full(torch.cat(full_free_group))
