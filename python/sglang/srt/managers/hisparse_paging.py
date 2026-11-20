from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from sglang.kernels.ops.kvcache import hisparse_paging as paging_kernels
from sglang.srt.environ import envs
from sglang.srt.layers.attention.graph_variants import HISPARSE_RESIDENT
from sglang.srt.model_executor.runner_utils.capture_mode import (
    get_capture_attention_variant,
)
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.schedule_batch import Req

device_module = get_device_module()

logger = logging.getLogger(__name__)


class HiSparsePagingManager:
    """Where the pages of resident HiSparse requests live.

    A resident request keeps its KV in the device pool. Its whole 64-token pages
    below the one being written may leave the pool: the allocator's map then names
    the page's host rows, which attention reads in place through the pool's mapped
    address range. On top of that home placement each group of layers that shares
    one selection keeps the pages it reads most in device frames of its own, and
    the all-resident decode graph keeps selected host rows in a row cache.
    """

    def __init__(self, coordinator: HiSparseCoordinator):
        self._coordinator = coordinator
        device = coordinator.device
        P = coordinator.page_size
        va = coordinator.mem_pool_device._hs_va
        self.host_row0 = int(va.host_row0(0))
        if self.host_row0 % P or any(
            int(va.host_row0(i)) != self.host_row0 for i in range(va.layer_num)
        ):
            raise ValueError(
                "HiSparse paged: the layers' host rows must start at one page-aligned row index"
            )
        self.num_device_pages = (
            coordinator.hs_el["k_max"] + P
        ) // P  # frames of the band too
        self._host_pages = va.host_rows // P
        if (self.host_row0 + va.host_rows) >= 2**31:
            raise RuntimeError("HiSparse paged: row indices past int32")
        log_pages = (coordinator.token_to_kv_pool_allocator.size_full + P) // P + 1
        self.frame_lp = torch.full(
            (self.num_device_pages,), -1, dtype=torch.int64, device=device
        )  # frame -> logical page
        self.evictable = torch.zeros(
            self.num_device_pages, dtype=torch.bool, device=device
        )
        self.score = torch.zeros(
            self.num_device_pages, dtype=torch.float32, device=device
        )
        self.host_first = torch.full(
            (log_pages,), -1, dtype=torch.int64, device=device
        )  # logical page -> host slot
        self._arange = torch.arange(P, dtype=torch.int64, device=device)
        self.registered_len = {}  # req_pool_idx -> positions registered for eviction (a page multiple)
        self.num_evictable_pages = 0
        self._last_step = 0
        self._steps = 0  # decode steps (the scores' clock)
        self._sel = torch.zeros(
            2, dtype=torch.int64, device=device
        )  # selected rows in HBM / on host, running
        self._sel_logged = (0, 0)
        self.stats = [
            0,
            0,
            0,
            0,
            0,
        ]  # evictions, pages evicted, seals, tokens sealed, failed reclaims
        self.evict_batch_pages = max(
            1, int(envs.SGLANG_HISPARSE_PAGED_EVICT_PAGES.get())
        )
        self._half = max(1, int(envs.SGLANG_HISPARSE_PAGED_HALF_LIFE.get()))
        self._check = bool(envs.SGLANG_HISPARSE_PAGED_CHECK.get())
        self._n_onhost = 0  # registered pages whose home placement is on host
        # Exclusive placement: a registered page in the device pool may hold no host
        # slot (xframe); it gets one, with a copy, before it can leave the pool.
        self.exclusive = bool(envs.SGLANG_HISPARSE_PAGED_EXCLUSIVE.get())
        self.num_exclusive_pages = 0
        self.host_len = {}  # req_pool_idx -> positions staged to host (a page multiple)
        # pages sealed for eviction, host slots given up, of those after a copy back,
        # pages promoted, frames moved in band take-backs, staging copies skipped,
        # frames moved up into the band
        self.xstats = [0, 0, 0, 0, 0, 0, 0]
        if self.exclusive:
            self.xframe = torch.zeros(
                self.num_device_pages, dtype=torch.bool, device=device
            )
            # logical page -> request slot and page index in the request
            self.lp_rid = torch.zeros(log_pages, dtype=torch.int64, device=device)
            self.lp_pos = torch.zeros(log_pages, dtype=torch.int64, device=device)
        self._stream = device_module.Stream()
        self._tick_steps = max(1, int(envs.SGLANG_HISPARSE_PAGED_TICK_STEPS.get()))
        self._hyst = float(envs.SGLANG_HISPARSE_PAGED_HYST.get())
        self._min = float(envs.SGLANG_HISPARSE_PAGED_MIN_SCORE.get())
        coordinator.token_to_kv_pool_allocator.host_row0 = self.host_row0
        logger.info(
            "HiSparse paged: %d HBM pages, %d host pages, host rows from index %d; evictions of >= %d pages, "
            "score half-life %d steps",
            self.num_device_pages,
            self._host_pages,
            self.host_row0,
            self.evict_batch_pages,
            self._half,
        )

        self._gswap = int(
            envs.SGLANG_HISPARSE_PAGED_GROUP_PAGES.get()
        )  # pages moved per group per tick
        if self._gswap < 1:
            raise RuntimeError(
                "HiSparse paged: SGLANG_HISPARSE_PAGED_GROUP_PAGES must be at least 1"
            )
        self._layer_group, g = [], -1
        for shared in coordinator._is_shared_index_layer:
            g += 0 if shared else 1
            self._layer_group.append(g)
        G = g + 1
        D, H = self.num_device_pages, self._host_pages
        self._lmax = max(self._layer_group.count(x) for x in range(G))
        base = [int(x) for x in coordinator.mem_pool_device.data_ptrs.cpu().numpy()]
        ptrs = np.zeros((G, self._lmax), dtype=np.uint64)
        fill = [0] * G
        for layer, gg in enumerate(self._layer_group):
            ptrs[gg, fill[gg]] = base[layer]
            fill[gg] += 1
        self.gptrs = torch.from_numpy(ptrs).to(device)
        self.gredir = torch.full((G, D + H + 1), -1, dtype=torch.int32, device=device)
        self.gcnt = torch.zeros((G, D + H), dtype=torch.int32, device=device)
        self.gscore = torch.zeros((G, D + H + 1), dtype=torch.float32, device=device)
        self.gfr = torch.full((G, D + 1), -2, dtype=torch.int32, device=device)
        self.ghost = torch.zeros((G, H + 1), dtype=torch.bool, device=device)
        self._grows = torch.arange(G, device=device)[:, None]
        self._gpending = None
        self._gmemo = None
        self.gstats = [
            0,
            0,
            0,
        ]  # ticks, pages moved (as of the last log line), ticks behind a running copy
        self._gmoved = torch.zeros((), dtype=torch.int64, device=device)
        self._gmoved_pin = torch.ones(
            1, dtype=torch.int64
        ).pin_memory()  # last tick's moves, read without a sync
        self._gevery = self._tick_steps
        self._gnext = 0
        self._ggrp = None
        # ---- host-row cache of the all-resident graph ----
        self._rc_rows = 0
        self._rc_memo = None
        rc = max(0, int(envs.SGLANG_HISPARSE_PAGED_ROWCACHE_ROWS.get()))
        if rc and coordinator.enable_prefetch:
            if rc & (rc - 1) or rc < 1024:
                raise RuntimeError(
                    "HiSparse paged: SGLANG_HISPARSE_PAGED_ROWCACHE_ROWS must be a power of two >= 1024"
                )
            if int(coordinator.mem_pool_device.bytes_per_token) % 8:
                raise RuntimeError(
                    "HiSparse paged: the row cache copies 8-byte words; the KV row size is not a multiple of 8"
                )
            alloc_ = coordinator.token_to_kv_pool_allocator
            # like a buffer, the scratch rows are not reachable through the logical map: below the band
            top = coordinator.hs_el["k_min"] + coordinator.page_size
            scr = alloc_.hs_alloc_below(rc, top)
            if scr is None:
                raise RuntimeError(
                    f"HiSparse paged: no room for {rc} row-cache rows in the device pool"
                )
            if int(scr.max()) >= top or int(scr.min()) < coordinator.page_size:
                raise RuntimeError(
                    "HiSparse paged: row-cache rows outside the static device pool"
                )
            alloc_.hs_keep_reserved(
                scr
            )  # not handed out again after a clear() (cache flush)
            self._rc_rows = rc
            self._rc_logc = rc.bit_length() - 1
            self._rc_blocks = max(
                1, int(envs.SGLANG_HISPARSE_PAGED_ROWCACHE_BLOCKS.get())
            )
            self._rc_cap = max(
                1024, min(rc, int(envs.SGLANG_HISPARSE_PAGED_ROWCACHE_LIST.get()))
            )
            self._rc_host_end = (
                self.host_row0 + self._host_pages * coordinator.page_size
            )
            i32 = dict(dtype=torch.int32, device=device)
            self.rc_scr = scr.to(torch.int32).contiguous()
            self.rc_tag = torch.full(
                (G, rc), -1, **i32
            )  # host row (index in the host pool) held by a slot
            self.rc_tep = torch.zeros(
                (G, rc), **i32
            )  # its host page's epoch at install
            self.rc_epoch = torch.ones(
                self._host_pages + 1, **i32
            )  # bumped when a host page dies
            self.rc_used = torch.zeros(rc, **i32)  # plan stamps: slot read as a hit
            self.rc_cval = torch.zeros(rc, **i32)  # plan: claimed by this host row
            self.rc_cst = torch.zeros(rc, **i32)  # plan stamps: slot claimed
            self._rc_stamp = torch.zeros(1, **i32)
            self.rc_src = torch.zeros(
                (G, self._rc_cap), **i32
            )  # install lists, one per group
            self.rc_dst = torch.zeros((G, self._rc_cap), **i32)
            self.rc_total = torch.zeros((G, 1), **i32)
            self.rc_stat = torch.zeros(
                3, dtype=torch.int64, device=device
            )  # rows: hits, installed, read in place
            self._rc_logged = (0, 0, 0)
            fill = [0] * G
            self._layer_k = []  # a layer's column in gptrs
            for layer, gg in enumerate(self._layer_group):
                self._layer_k.append(gg * self._lmax + fill[gg])
                fill[gg] += 1
            # load the kernels now (HBM has room): a plan over one slot that is not a host row, an empty install
            z1 = torch.full((1,), -1, dtype=torch.int32, device=device)
            self._rc_plan(z1, 0)
            self._rc_copy(0, 0)
            self._rc_copy(0, 0, inline=True)
            self.rc_stat.zero_()
            logger.info(
                "HiSparse paged: host-row cache for the all-resident graph: %d scratch rows (%d device pages) per layer "
                "group, direct-mapped; install lists of %d rows; copy kernel with %d programs; tables %.1f MiB",
                rc,
                rc // coordinator.page_size,
                self._rc_cap,
                self._rc_blocks,
                (
                    self.rc_tag.numel() * 8
                    + rc * 16
                    + self.rc_src.numel() * 8
                    + self.rc_epoch.numel() * 4
                )
                / 2**20,
            )
        # every kernel paging launches later is loaded now, while HBM has room (a Triton kernel first loaded with
        # HBM full fails: "Triton Error [CUDA]: out of memory"): a redirect of slot -1, a copy of the padding page 0
        # onto itself
        z = torch.full((1,), -1, dtype=torch.int32, device=device)
        self._redirect_raw(z, 0, torch.empty_like(z))
        z64 = torch.zeros(1, dtype=torch.int64, device=device)
        row = int(coordinator.mem_pool_device.bytes_per_token)
        paging_kernels.copy_group_pages[
            (1, self._lmax, (coordinator.page_size * row + 4095) // 4096)
        ](
            self.gptrs,
            z64,
            z64,
            z64,
            torch.ones(1, dtype=torch.int32, device=device),
            LMAX=self._lmax,
            PAGE_BYTES=coordinator.page_size * row,
            ROW=row,
            TILE=4096,
        )
        logger.info(
            "HiSparse paged: per-group placement, %d groups of up to %d layers, up to %d pages moved per group "
            "every %d steps; tables %.1f MiB",
            G,
            self._lmax,
            self._gswap,
            self._tick_steps,
            (
                self.gredir.numel() * 4
                + self.gcnt.numel() * 4
                + self.gscore.numel() * 4
                + self.gfr.numel() * 4
                + self.ghost.numel()
            )
            / 2**20,
        )

    def step(self) -> None:
        """One decode or verify step: the scores' clock, and the group moves it paces."""
        self._steps += 1
        if self._coordinator._resident:
            self._gtick()
        if self._steps % 2000 == 0:
            self._fold()
            self._log()

    def sealable_end(self, req: Req) -> int:
        """Positions [0, end) are whole pages below the page the request is writing."""
        coordinator = self._coordinator
        P = coordinator.page_size
        if coordinator.hs_spec_res:  # MTP: positions at or above the committed length may still be rewritten by a verify
            return int(req.kv.kv_committed_len) // P * P
        return (int(req.kv.kv_allocated_len) - 1) // P * P

    def register(self, rid: int, req: Req) -> None:
        """Make the sealed full pages [registered, end) of resident request `rid` evictable. They are in HBM: a page
        leaves HBM only after it was registered."""
        coordinator = self._coordinator
        P = coordinator.page_size
        start = self.registered_len[rid]
        if self.exclusive:
            self._register_exclusive(rid, req)
            return
        end = min(coordinator._hs_staged_len[rid], self.sealable_end(req)) // P * P
        if end <= start:
            return
        mapping = (
            coordinator.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
        )
        first = coordinator.req_to_token_pool.req_to_token[rid, start:end:P].to(
            torch.int64
        )
        dev_first = mapping[first]
        host_first = coordinator.req_to_host_pool[rid, start:end:P]
        if self._check:
            last = coordinator.req_to_token_pool.req_to_token[
                rid, start + P - 1 : end : P
            ].to(torch.int64)
            ok = (
                bool((first % P == 0).all())
                and bool((last - first == P - 1).all())
                and bool((dev_first % P == 0).all())
                and bool((dev_first >= P).all())
                and bool((dev_first < self.host_row0).all())
                and bool((mapping[last] - dev_first == P - 1).all())
                and bool((host_first % P == 0).all())
                and bool((host_first >= 0).all())
                and bool(
                    (
                        coordinator.req_to_host_pool[rid, start + P - 1 : end : P]
                        - host_first
                        == P - 1
                    ).all()
                )
                and not bool(self.evictable[dev_first // P].any())
            )
            if not ok:
                raise RuntimeError(
                    f"HiSparse paged: pages of request slot {rid} [{start}, {end}) are not aligned"
                )
        frames = dev_first // P
        prior = (self.score * self.evictable).sum() / self.evictable.sum().clamp(min=1)
        self.frame_lp[frames] = first // P
        self.host_first[first // P] = host_first
        self.score[frames] = (
            prior  # a new page starts at the mean score, not as the coldest
        )
        D = self.num_device_pages
        ev = self.evictable[None, :]
        gprior = (self.gscore[:, :D] * ev).sum(1) / ev.sum().clamp(
            min=1
        )  # the same rule in every group
        self.gscore[:, frames] = gprior[:, None]
        self.gscore[:, D + host_first // P] = 0
        self.evictable[frames] = True
        self.registered_len[rid] = end
        self.num_evictable_pages += (end - start) // P

    # ---- exclusive placement -------------------------------------------------------------------------------------------
    # A registered page is in one of three states: in a device frame with no host
    # slot (xframe), in a device frame with a host copy (evictable), or on host
    # (the map names its host rows). A host slot is released only while the page's
    # rows are complete in its frame for every layer group; a frame is freed only
    # once the host copy was launched ahead of every later reader on the streams.
    def _register_exclusive(self, rid: int, req: Req) -> None:
        """Track the whole pages [registered, end) of resident request `rid`: those staged to host as evictable, the
        others as exclusive. They are in HBM: a page leaves HBM only after it was registered."""
        coordinator = self._coordinator
        P = coordinator.page_size
        start = self.registered_len[rid]
        end = self.sealable_end(req) // P * P
        if end <= start:
            return
        mapping = (
            coordinator.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
        )
        first = coordinator.req_to_token_pool.req_to_token[rid, start:end:P].to(
            torch.int64
        )
        dev_first = mapping[first]
        # Positions below host_len were staged; nothing else of an unregistered page has a host slot.
        num_staged = max(0, min(end, self.host_len.get(rid, 0)) - start) // P
        num_pages = (end - start) // P
        host_first = coordinator.req_to_host_pool[rid, start:end:P]
        if self._check:
            last = coordinator.req_to_token_pool.req_to_token[
                rid, start + P - 1 : end : P
            ].to(torch.int64)
            staged = host_first[:num_staged]
            ok = (
                bool((first % P == 0).all())
                and bool((last - first == P - 1).all())
                and bool((dev_first % P == 0).all())
                and bool((dev_first >= P).all())
                and bool((dev_first < self.host_row0).all())
                and bool((mapping[last] - dev_first == P - 1).all())
                and bool((staged % P == 0).all())
                and bool((staged >= 0).all())
                and bool((host_first[num_staged:] == -1).all())
                and bool(
                    (
                        coordinator.req_to_host_pool[
                            rid, start + P - 1 : start + num_staged * P : P
                        ]
                        - staged
                        == P - 1
                    ).all()
                )
                and not bool(self.evictable[dev_first // P].any())
                and not bool(self.xframe[dev_first // P].any())
            )
            if not ok:
                raise RuntimeError(
                    f"HiSparse paged: pages of request slot {rid} [{start}, {end}) are not aligned"
                )
        frames = dev_first // P
        lp = first // P
        tracked = self.evictable | self.xframe
        prior = (self.score * tracked).sum() / tracked.sum().clamp(min=1)
        self.frame_lp[frames] = lp
        self.host_first[lp] = host_first
        self.lp_rid[lp] = rid
        self.lp_pos[lp] = torch.arange(
            start // P, end // P, dtype=torch.int64, device=lp.device
        )
        self.score[frames] = prior
        D = self.num_device_pages
        tr = tracked[None, :]
        gprior = (self.gscore[:, :D] * tr).sum(1) / tr.sum().clamp(min=1)
        self.gscore[:, frames] = gprior[:, None]
        self.gscore[:, D + host_first[:num_staged] // P] = 0
        self.evictable[frames[:num_staged]] = True
        self.xframe[frames[num_staged:]] = True
        self.registered_len[rid] = end
        self.num_evictable_pages += num_staged
        self.num_exclusive_pages += num_pages - num_staged

    def track_all(self) -> None:
        """Register every resident request's pages that became whole pages below its writing position."""
        for rid, req in self._coordinator._resident.items():
            if self.sealable_end(req) > self.registered_len[rid]:
                self._register_exclusive(rid, req)

    def untracked_sealable_pages(self) -> int:
        return (
            sum(
                max(self.sealable_end(req) - self.registered_len.get(rid, 0), 0)
                for rid, req in self._coordinator._resident.items()
            )
            // self._coordinator.page_size
        )

    def _page_slots(self, first: torch.Tensor) -> torch.Tensor:
        """[pages * P] slots of the pages that start at `first`."""
        return (first[:, None] + self._arange[None, :]).reshape(-1)

    def _set_host_slots(self, lp: torch.Tensor, host_first: torch.Tensor) -> None:
        """Record the host slots (-1: none) of registered logical pages `lp`."""
        coordinator = self._coordinator
        P = coordinator.page_size
        rid, pos = self.lp_rid[lp], self.lp_pos[lp]
        slots = torch.where(
            host_first[:, None] >= 0,
            host_first[:, None] + self._arange[None, :],
            torch.full_like(self._arange[None, :], -1),
        )
        coordinator.req_to_host_pool[
            rid[:, None], pos[:, None] * P + self._arange[None, :]
        ] = slots
        self.host_first[lp] = host_first

    def _copy_pages(self, src_first: torch.Tensor, dst_first: torch.Tensor) -> None:
        """Rows [src, src + P) -> [dst, dst + P) of every layer, device or host rows alike. Current stream."""
        coordinator = self._coordinator
        P = coordinator.page_size
        k = int(src_first.numel())
        if k == 0:
            return
        G = self.gptrs.shape[0]
        grp = torch.arange(G, device=src_first.device).repeat_interleave(k).contiguous()
        src = src_first.to(torch.int64).repeat(G).contiguous()
        dst = dst_first.to(torch.int64).repeat(G).contiguous()
        ok = torch.ones(G * k, dtype=torch.int32, device=src_first.device)
        row = int(coordinator.mem_pool_device.bytes_per_token)
        tile = 4096
        paging_kernels.copy_group_pages[
            (G * k, self._lmax, (P * row + tile - 1) // tile)
        ](
            self.gptrs,
            grp,
            src,
            dst,
            ok,
            LMAX=self._lmax,
            PAGE_BYTES=P * row,
            ROW=row,
            TILE=tile,
        )

    def seal_exclusive(
        self, n_pages: int, lo_frame: int = 0, hi_frame: int = -1, frames=None
    ) -> int:
        """The n coldest exclusive pages (of frames [lo_frame, hi_frame), or the first n of `frames`) get a host
        slot and a copy of their rows there: they become evictable. On the current stream: the caller ordered it
        after the in-flight batch."""
        coordinator = self._coordinator
        P = coordinator.page_size
        host = coordinator.mem_pool_host
        k = min(int(n_pages), self.num_exclusive_pages, host.available_size() // P)
        if k <= 0:
            return 0
        if frames is not None:
            sel = frames[:k]
        else:
            self._fold()
            cand = self.xframe
            if lo_frame > 0 or hi_frame >= 0:
                fr = torch.arange(cand.numel(), device=cand.device)
                cand = cand & (fr >= lo_frame)
                if hi_frame >= 0:
                    cand = cand & (fr < hi_frame)
                k = min(k, int(cand.sum()))
                if k <= 0:
                    return 0
            score = torch.where(
                cand, self.score, torch.full_like(self.score, float("inf"))
            )
            sel = torch.topk(score, k, largest=False).indices
        lp = self.frame_lp[sel]
        host_locs = host.alloc(k * P).to(coordinator.device)
        host_first = host_locs.view(k, P)[:, 0].contiguous()
        if self._check and (
            bool((host_first % P != 0).any())
            or bool((host_locs.view(k, P) - host_first[:, None] != self._arange).any())
            or bool((self.host_first[lp] != -1).any())
        ):
            raise RuntimeError(
                "HiSparse paged: host slots of a sealed page are not one page"
            )
        host.backup_from_device_all_layer(
            coordinator.mem_pool_device,
            host_locs,
            self._page_slots(sel * P),
            io_backend="kernel",
        )
        self._set_host_slots(lp, host_first)
        self.gscore[:, self.num_device_pages + host_first // P] = 0
        self.xframe[sel] = False
        self.evictable[sel] = True
        self.num_exclusive_pages -= k
        self.num_evictable_pages += k
        self.stats[2] += 1
        self.stats[3] += k * P
        self.xstats[0] += k
        return k

    def _seal_coldest(self, n_pages: int) -> None:
        """Before an eviction of n pages: the exclusive pages among the n coldest registered pages get a host slot
        and their copy, so that the coldest pages leave whatever their state. The slots are free ones first, then
        those of the hottest pages that have one (they stay in HBM, exclusive from then on)."""
        coordinator = self._coordinator
        self.track_all()
        k = min(int(n_pages), self.num_evictable_pages + self.num_exclusive_pages)
        if k <= 0 or self.num_exclusive_pages <= 0:
            return
        self._gfinish(block=True)
        self._fold()
        cand = self.evictable | self.xframe
        cold = torch.topk(
            torch.where(cand, self.score, torch.full_like(self.score, float("inf"))),
            k,
            largest=False,
        ).indices
        exclusive = cold[self.xframe[cold]]  # coldest first
        num_exclusive = int(exclusive.numel())
        if num_exclusive <= 0:
            return
        free_pages = coordinator.mem_pool_host.available_size() // coordinator.page_size
        if free_pages < num_exclusive:
            self.give_up_host(num_exclusive - free_pages, keep=cold)
        self.seal_exclusive(num_exclusive, frames=exclusive)

    def _restore_home(self, n_pages: int) -> None:
        """The n hottest evictable frames that some layer group does not read its home page from get that page's
        rows back from host for those groups (guests there go back to host). Current stream, after the in-flight
        batch and with no group move in flight."""
        coordinator = self._coordinator
        P, D, H = coordinator.page_size, self.num_device_pages, self._host_pages
        away_any = self.evictable & (self.gfr[:, :D] != -2).any(0)
        k = min(int(n_pages), int(away_any.sum()))
        if k <= 0:
            return
        fr = torch.topk(
            torch.where(
                away_any, self.score, torch.full_like(self.score, -float("inf"))
            ),
            k,
        ).indices
        rows = self._grows
        st = self.gfr[:, fr].to(torch.int64)  # [G, k]
        away, guest = st >= -1, st >= 0
        home_hp = self.host_first[self.frame_lp[fr]] // P  # [k]
        gs = self.gscore[:, fr]
        # guests there go back to host
        self.gredir[rows, torch.where(guest, D + st, D + H)] = -1
        self.ghost[rows, torch.where(guest, st, H)] = True
        self.gscore[rows, torch.where(guest, D + st, D + H)] = gs
        # the home page's rows of those groups: host -> frame
        g_idx, f_idx = torch.nonzero(away, as_tuple=True)
        n = int(g_idx.numel())
        row = int(coordinator.mem_pool_device.bytes_per_token)
        tile = 4096
        paging_kernels.copy_group_pages[(n, self._lmax, (P * row + tile - 1) // tile)](
            self.gptrs,
            g_idx.contiguous(),
            (self.host_row0 + home_hp[f_idx] * P).contiguous(),
            (fr[f_idx] * P).contiguous(),
            torch.ones(n, dtype=torch.int32, device=fr.device),
            LMAX=self._lmax,
            PAGE_BYTES=P * row,
            ROW=row,
            TILE=tile,
        )
        own = self.gscore[rows, D + home_hp[None, :]]
        self.gscore[rows, torch.where(away, fr[None, :], D + H)] = own
        self.gscore[rows, torch.where(away, D + home_hp[None, :], D + H)] = 0
        self.gredir[:, fr] = -1
        self.gfr[:, fr] = -2
        self.ghost[:, H] = False
        self.xstats[2] += k

    def give_up_host(self, n_pages: int, restore: bool = False, keep=None) -> int:
        """The n hottest evictable pages that every layer group reads from their frame give their host slots back
        and become exclusive. restore: also pages some group reads from host, after copying their rows back.
        keep: frames that stay as they are. The caller ordered the current stream after the in-flight batch (it
        may read those host rows)."""
        coordinator = self._coordinator
        P, D = coordinator.page_size, self.num_device_pages
        k = min(int(n_pages), self.num_evictable_pages)
        if k <= 0:
            return 0
        self._gfinish(block=True)
        self._fold()
        cand = self.evictable & (self.gfr[:, :D] == -2).all(0)
        if keep is not None:
            cand[keep] = False
        n = int(cand.sum())
        if n < k and restore:
            self._restore_home(k - n)
            cand = self.evictable & (self.gfr[:, :D] == -2).all(0)
            n = int(cand.sum())
        k = min(k, n)
        if k <= 0:
            return 0
        sel = torch.topk(
            torch.where(cand, self.score, torch.full_like(self.score, -float("inf"))), k
        ).indices
        lp = self.frame_lp[sel]
        host_first = self.host_first[lp]
        if self._rc_rows:
            self.rc_epoch[host_first // P] += 1
        self.gscore[:, D + host_first // P] = 0
        self._set_host_slots(lp, torch.full_like(host_first, -1))
        self.evictable[sel] = False
        self.xframe[sel] = True
        self.num_evictable_pages -= k
        self.num_exclusive_pages += k
        coordinator.mem_pool_host.free(self._page_slots(host_first))
        self.xstats[1] += k
        return k

    def promote(self, n_pages: int) -> int:
        """The n hottest pages on host move into free device frames (every layer) and give their host slots back.
        Current stream; the caller ordered it after the in-flight batch."""
        coordinator = self._coordinator
        P, D, H = coordinator.page_size, self.num_device_pages, self._host_pages
        alloc = coordinator.token_to_kv_pool_allocator
        dev = alloc.hisparse_attn_allocator
        k = min(int(n_pages), self._n_onhost, dev.available_size() // P)
        if k <= 0:
            return 0
        self._gfinish(block=True)
        self._fold()
        mapping = alloc.full_to_hisparse_device_index_mapping
        reg = torch.nonzero(self.host_first >= 0).squeeze(1)
        reg = reg[mapping[reg * P] >= self.host_row0]  # registered pages on host
        k = min(k, int(reg.numel()))
        if k <= 0:
            return 0
        hp_all = self.host_first[reg] // P
        lp = reg[torch.topk(self.gscore[:, D + hp_all].sum(0), k).indices]
        host_first = self.host_first[lp]
        hp = host_first // P
        new = dev.alloc(k * P)
        if new is None:
            return 0
        frames = new.view(k, P)[:, 0].to(torch.int64) // P
        self._copy_pages(self.host_row0 + host_first, frames * P)
        # the page's guests: their frames become holes; its score per group follows it into the frame
        rows = self._grows
        r = self.gredir[:, D + hp].to(torch.int64)  # [G, k]
        isg = r >= 0
        sc = torch.where(isg, self.gscore[rows, r.clamp(min=0)], self.gscore[:, D + hp])
        self.gfr[rows, torch.where(isg, r, D)] = -1
        self.gscore[rows, torch.where(isg, r, D + H)] = 0
        self.gredir[:, D + hp] = -1
        self.ghost[:, hp] = False
        self.gscore[:, D + hp] = 0
        self.gscore[:, frames] = sc
        self.score[frames] = sc.sum(0)
        self.gfr[:, D] = -2
        if self._rc_rows:
            self.rc_epoch[hp] += 1
        mapping[self._page_slots(lp * P)] = self._page_slots(frames * P)
        self.frame_lp[frames] = lp
        self.xframe[frames] = True
        self._set_host_slots(lp, torch.full_like(host_first, -1))
        self._n_onhost -= k
        self.num_exclusive_pages += k
        coordinator._hs_pt_memo = None
        coordinator.mem_pool_host.free(self._page_slots(host_first))
        self.xstats[3] += k
        return k

    def ensure_host(self, need_slots: int, must: bool) -> bool:
        """Make `need_slots` host slots free: pages in the device pool give theirs back; when the slots must be had
        (a buffered request's host rows), also after copying rows back into frames, and by moving pages from host
        into free frames."""
        coordinator = self._coordinator
        short = need_slots - coordinator.mem_pool_host.available_size()
        if short <= 0:
            return True
        pages = -(-short // coordinator.page_size)
        # The in-flight batch may read the host rows given back here; whatever writes
        # them next is ordered after this stream.
        if coordinator.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(
                coordinator.decode_producer_stream
            )
        coordinator.wait_for_pending_backup()
        pages -= self.give_up_host(pages)
        if pages > 0 and must:
            pages -= self.give_up_host(pages, restore=True)
        if pages > 0 and must:
            pages -= self.promote(pages)
        return pages <= 0

    def relocate(self, old_frames: torch.Tensor, new_frames: torch.Tensor) -> None:
        """The pages of `old_frames` were copied to `new_frames` and the map rewritten: the exclusive ones keep
        their registration (evictable pages never get here: they leave by eviction)."""
        x = self.xframe[old_frames]
        old, new = old_frames[x], new_frames[x]
        self.frame_lp[new] = self.frame_lp[old]
        self.score[new] = self.score[old]
        self.gscore[:, new] = self.gscore[:, old]
        self.xframe[new] = True
        self.frame_lp[old] = -1
        self.score[old] = 0
        self.gscore[:, old] = 0
        self.xframe[old] = False

    def lift(self, n_pages: int, floor_frame: int) -> int:
        """Exclusive pages below `floor_frame` move to free frames at or above it (every layer; current stream, after
        the in-flight batch). Room below the band when nothing there can leave for host."""
        coordinator = self._coordinator
        P = coordinator.page_size
        alloc = coordinator.token_to_kv_pool_allocator
        fr = torch.arange(self.xframe.numel(), device=self.xframe.device)
        cand = self.xframe & (fr < floor_frame)
        k = min(int(n_pages), int(cand.sum()))
        if k <= 0:
            return 0
        new = alloc.hs_alloc_above(k * P, floor_frame * P)
        if new is None:
            return 0
        self._fold()
        old = torch.topk(
            torch.where(cand, self.score, torch.full_like(self.score, float("inf"))),
            k,
            largest=False,
        ).indices
        frames = new.view(k, P)[:, 0].to(torch.int64) // P
        self._copy_pages(old * P, frames * P)
        lp = self.frame_lp[old]
        mapping = alloc.full_to_hisparse_device_index_mapping
        mapping[self._page_slots(lp * P)] = self._page_slots(frames * P)
        self.relocate(old, frames)
        self.xstats[6] += k
        alloc.free_hisparse_indices(self._page_slots(old * P))
        coordinator._hs_pt_memo = None
        return k

    def seal_all(self) -> None:
        """Copy every resident request's unsealed full pages to host (page units) and register them. On the current
        stream: the caller ordered it after the in-flight batch and the pending backup."""
        coordinator = self._coordinator
        mapping = (
            coordinator.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
        )
        for rid, req in coordinator._resident.items():
            staged = coordinator._hs_staged_len[rid]
            end = self.sealable_end(req)
            if end <= staged:
                self.register(
                    rid, req
                )  # staged pages that became whole pages below the writing position
                continue
            logical = coordinator.req_to_token_pool.req_to_token[rid, staged:end].to(
                torch.int64
            )
            device_locs = mapping[logical]
            host_locs = coordinator.mem_pool_host.alloc_paged_token_slots(
                coordinator.req_to_host_pool,
                coordinator.req_to_host_pool_allocated_len,
                rid,
                staged,
                end - staged,
            )
            coordinator.mem_pool_host.backup_from_device_all_layer(
                coordinator.mem_pool_device, host_locs, device_locs, io_backend="kernel"
            )
            coordinator._hs_staged_len[rid] = end
            self.stats[2] += 1
            self.stats[3] += end - staged
            self.register(rid, req)

    def _fold(self) -> None:
        """Fold the selection counters into the decayed scores of the HBM frames and the host pages."""
        D = self.num_device_pages
        steps = self._steps - self._last_step
        self._last_step = self._steps
        decay = 0.5 ** (steps / self._half)
        # the groups' counters; the allocator's view is their sum
        T = D + self._host_pages
        self.gscore[:, :T].mul_(decay).add_(self.gcnt.to(torch.float32))
        tot = self.gcnt.sum(0)
        self._sel += torch.stack(
            (tot[:D].sum(), tot[D:].sum())
        )  # rows selected in HBM, on host
        self.gcnt.zero_()
        self.score.copy_(self.gscore[:, :D].sum(0))

    # ---- paged residency host-row cache (all-resident graph) --------------------------------------------------------------
    def _rc_plan(self, slots: torch.Tensor, g: int):
        """One plan of group g over a slot table: (the indexer layer's table, the reusing layers' table). Leaves the
        group's install list and its length. Current stream; no host sync."""
        coordinator = self._coordinator
        K = paging_kernels
        flat = slots.reshape(-1)
        n = flat.numel()
        B = 1024
        nb = (n + B - 1) // B
        C, P = self._rc_rows, coordinator.page_size
        tag, tep, ep = self.rc_tag[g], self.rc_tep[g], self.rc_epoch
        h0, h1 = self.host_row0, self._rc_host_end
        self._rc_stamp.add_(1)
        st = self._rc_stamp
        K.row_cache_mark[(nb,)](
            flat,
            n,
            tag,
            tep,
            ep,
            self.rc_used,
            st,
            h0,
            h1,
            LOGC=self._rc_logc,
            PAGE=P,
            BLOCK=B,
        )
        K.row_cache_claim[(nb,)](
            flat,
            n,
            tag,
            tep,
            ep,
            self.rc_used,
            self.rc_cval,
            self.rc_cst,
            st,
            h0,
            h1,
            LOGC=self._rc_logc,
            PAGE=P,
            BLOCK=B,
        )
        cb = C // B
        cnt = torch.empty(cb, dtype=torch.int32, device=flat.device)
        K.row_cache_count[(cb,)](self.rc_cst, st, cnt, BLOCK=B)
        c64 = cnt.to(torch.int64)
        off = (torch.cumsum(c64, 0) - c64).contiguous()
        self.rc_total[g].copy_(
            c64.sum().clamp(max=self._rc_cap).to(torch.int32).reshape(1)
        )
        K.row_cache_compact[(cb,)](
            self.rc_cst,
            self.rc_cval,
            st,
            off,
            self.rc_scr,
            ep,
            tag,
            tep,
            self.rc_src[g],
            self.rc_dst[g],
            h0,
            self._rc_cap,
            PAGE=P,
            BLOCK=B,
        )
        out_a = torch.empty_like(flat)
        out_r = torch.empty_like(flat)
        K.row_cache_resolve[(nb,)](
            flat,
            out_a,
            out_r,
            n,
            tag,
            tep,
            ep,
            self.rc_used,
            self.rc_cval,
            self.rc_cst,
            st,
            self.rc_scr,
            self.rc_stat,
            h0,
            h1,
            LOGC=self._rc_logc,
            PAGE=P,
            BLOCK=B,
        )
        return out_a.view(slots.shape), out_r.view(slots.shape)

    def _rc_copy(self, g: int, layer_id: int, inline: bool = False) -> None:
        """Install group g's planned rows in `layer_id`. On the stream that is current. inline: the indexer layer's
        own copy, which nothing overlaps: four times the programs of the side-stream copies."""
        coordinator = self._coordinator
        row = int(coordinator.mem_pool_device.bytes_per_token)
        nb = self._rc_blocks * (4 if inline else 1)
        paging_kernels.row_cache_copy[(nb,)](
            self.gptrs,
            self._layer_k[layer_id],
            self.rc_src[g],
            self.rc_dst[g],
            self.rc_total[g],
            NB=nb,
            RPI=64,
            ROWW=row // 8,
            TILE=1 << (row // 8 - 1).bit_length(),
            num_warps=8,
        )

    def read_through_row_cache(self, red: torch.Tensor, layer_id: int) -> torch.Tensor:
        """All-resident graph: the slot table a layer reads, through its group's host-row cache."""
        coordinator = self._coordinator
        if not self._rc_rows or red.ndim not in (2, 3):
            return red
        if get_capture_attention_variant() != HISPARSE_RESIDENT:
            return red
        memo = self._rc_memo
        if coordinator._is_shared_index_layer[layer_id]:
            if memo is None or memo[0] is not red:
                return red  # no plan for this table: read in place
            coordinator._prefetch_events[coordinator._prefetch_slot[layer_id]].wait(
                device_module.current_stream()
            )
            return memo[2]
        if memo is not None and memo[0] is red:
            return memo[2]
        g = self._layer_group[layer_id]
        src = red if red.is_contiguous() else red.contiguous()
        out_a, out_r = self._rc_plan(src, g)
        group = coordinator._prefetch_groups.get(layer_id) or ()
        if group:  # the reusing layers' installs: after everything launched so far (the plan, the previous readers)
            coordinator.prefetch_stream.wait_stream(device_module.current_stream())
            with device_module.stream(coordinator.prefetch_stream):
                for skip_layer in group:
                    self._rc_copy(g, skip_layer)
                    coordinator._prefetch_events[
                        coordinator._prefetch_slot[skip_layer]
                    ].record(coordinator.prefetch_stream)
        self._rc_copy(
            g, layer_id, inline=True
        )  # the indexer layer's own misses, in line: then it reads them too
        self._rc_memo = (red, out_a, out_r)
        return out_r

    # ---- paged residency: per-group placement --------------------------------------------------------------------------
    # Per group g and HBM frame f, gfr[g, f]: -2 the frame holds its home page (the allocator's map); -1 a hole (the
    # group reads the home page from host: gredir[g, f] = its host page); >= 0 a guest (that host page's rows, the
    # group's layers only: gredir[g, DEV + host page] = f); -3 reserved for a copy in flight. ghost[g, host page]:
    # the page is on host for everyone and no guest in g (a promotion candidate). The last column of each table is a
    # sink for masked writes.
    def _redirect_raw(self, src: torch.Tensor, g: int, out: torch.Tensor) -> None:
        coordinator = self._coordinator
        n = src.numel()
        if n == 0:
            return
        assert (
            src.dtype == torch.int32
            and out.dtype == torch.int32
            and src.is_contiguous()
            and out.is_contiguous()
        )
        block = 1024
        D, H = self.num_device_pages, self._host_pages
        paging_kernels.redirect_slots[((n + block - 1) // block,)](
            src,
            out,
            n,
            self.gredir[g],
            self.gcnt[g],
            self.host_row0 // coordinator.page_size,
            D,
            D + H,
            PAGE=coordinator.page_size,
            BLOCK=block,
        )

    def redirect(self, locs: torch.Tensor, layer_id: int) -> torch.Tensor:
        """The slots this layer's attention reads: `locs` through its group's redirects. An anchor computes (and
        counts); its skip layers, handed the same slot table, reuse the anchor's result."""
        coordinator = self._coordinator
        g = self._layer_group[layer_id]
        key = (
            locs.data_ptr(),
            tuple(locs.shape),
            g,
            torch.cuda.is_current_stream_capturing(),
        )
        memo = self._gmemo
        if (
            coordinator._is_shared_index_layer[layer_id]
            and memo is not None
            and memo[0] == key
        ):
            return memo[1]
        src = locs if locs.is_contiguous() else locs.contiguous()
        out = torch.empty_like(src)
        self._redirect_raw(src, g, out)
        self._gmemo = (key, out)
        return out

    def _gdrop(self, frames: torch.Tensor, hp) -> None:
        """The allocator takes `frames` back (hp: their home pages' host pages when those stay alive on host, None when
        the pages die): every group kicks its guest there back to host and clears its redirects of the frame."""
        D, H = self.num_device_pages, self._host_pages
        rows = self._grows
        st = self.gfr[:, frames].to(torch.int64)  # [G, k]
        guest = st >= 0
        gs = self.gscore[:, frames]
        self.gredir[rows, torch.where(guest, D + st, D + H)] = -1
        self.ghost[rows, torch.where(guest, st, H)] = True
        self.gscore[rows, torch.where(guest, D + st, D + H)] = gs
        if (
            hp is not None
        ):  # the home pages are on host for every group now; where they were read in place the score moves
            self.gscore[rows, torch.where(st == -2, D + hp[None, :], D + H)] = gs
            self.ghost[:, hp] = True
        self.gredir[:, frames] = -1
        self.gfr[:, frames] = -2
        self.gscore[:, frames] = 0
        self.ghost[:, H] = False

    def _gfinish(self, block: bool) -> bool:
        """Complete the group moves whose copies have finished: the pages are read from their frames from now on.
        No host sync unless block (a moved pair is masked by ok; masked writes go to the sink columns)."""
        if self._gpending is None:
            return True
        ev, fr, hp, back, ok = self._gpending[:5]
        if not ev.query():
            if not block:
                return False
            ev.synchronize()
        D, H = self.num_device_pages, self._host_pages
        T = D + H
        rows = self._grows
        sc = self.gscore[rows, D + hp]
        # a home page back in its own frame: the frame's redirect goes; a guest: its host page points at the frame
        self.gredir[
            rows, torch.where(ok, torch.where(back, fr, D + hp), torch.full_like(fr, T))
        ] = torch.where(back, torch.full_like(fr, -1), fr).to(torch.int32)
        self.gfr[rows, torch.where(ok, fr, torch.full_like(fr, D))] = torch.where(
            back, torch.full_like(hp, -2), hp
        ).to(torch.int32)
        self.gscore[rows, torch.where(ok, fr, torch.full_like(fr, T))] = sc
        self.gscore[rows, torch.where(ok, D + hp, torch.full_like(hp, T))] = 0
        self.gfr[:, D] = -2
        self._gpending = None
        return True

    def _gtick(self) -> None:
        """Per-group moves, at most gswap of each kind per group: (B) a home page the group reads from host returns
        to its own frame when it is hotter than what sits there; (A) the hottest pages that are on host for everyone
        become guests in the group's coldest frames. No host sync: every candidate pair carries an ok flag, masked
        writes go to the sink columns, the copy kernel skips masked pairs. The tick backs off while nothing moves."""
        coordinator = self._coordinator
        if self._steps < self._gnext:
            return
        if not self._gfinish(block=False):
            self.gstats[2] += 1
            return
        moved = int(
            self._gmoved_pin[0]
        )  # the previous tick's count (copied without blocking; may lag one tick)
        self._gevery = self._tick_steps if moved > 0 else min(self._gevery * 2, 512)
        self._gnext = self._steps + self._gevery
        if self._n_onhost <= 0 or self.num_evictable_pages <= 0:
            return
        P, D, H = coordinator.page_size, self.num_device_pages, self._host_pages
        T = D + H
        m = min(self._gswap, D, H)
        self._fold()
        sc = self.gscore
        inf = float("inf")
        rows = self._grows
        home_hp = (self.host_first[self.frame_lp.clamp(min=0)] // P).clamp(
            min=0, max=H - 1
        )  # [D]
        sinkT = lambda mask, idx: torch.where(mask, idx, torch.full_like(idx, T))
        sinkH = lambda mask, idx: torch.where(mask, idx, torch.full_like(idx, H))
        sinkD = lambda mask, idx: torch.where(mask, idx, torch.full_like(idx, D))
        # (B) home pages back into their own frames (holes or guests there)
        st = self.gfr[:, :D].to(torch.int64)
        away = st >= -1
        own = torch.where(
            away,
            sc[:, D:T].gather(1, home_hp[None, :].expand(st.shape[0], -1)),
            torch.full_like(sc[:, :D], -inf),
        )
        occ = torch.where(st == -1, torch.full_like(sc[:, :D], -1.0), sc[:, :D])
        top = torch.topk(
            torch.where(
                own > self._hyst * occ.clamp(min=0), own, torch.full_like(own, -inf)
            ),
            m,
            dim=1,
        )
        okb = top.values >= self._min
        fb = top.indices
        sb = st.gather(1, fb)
        isg = okb & (sb >= 0)  # the guest there goes back to host
        gsc = sc.gather(1, fb)
        self.gredir[rows, sinkT(isg, D + sb)] = -1
        self.ghost[rows, sinkH(isg, sb)] = True
        self.gscore[rows, sinkT(isg, D + sb)] = gsc
        self.gfr[rows, sinkD(okb, fb)] = -3
        # (A) pages on host for everyone into the coldest eligible frames
        st = self.gfr[:, :D].to(torch.int64)
        hot = torch.topk(
            torch.where(
                self.ghost[:, :H], sc[:, D:T], torch.full_like(sc[:, D:T], -inf)
            ),
            m,
            dim=1,
        )
        elig = (self.evictable[None, :] & (st == -2)) | (st >= -1)
        cs = torch.where(
            elig,
            torch.where(st == -1, torch.full_like(sc[:, :D], -1.0), sc[:, :D]),
            torch.full_like(sc[:, :D], inf),
        )
        cold = torch.topk(cs, m, dim=1, largest=False)
        oka = (
            (hot.values > self._hyst * cold.values.clamp(min=0))
            & (hot.values >= self._min)
            & (cold.values < inf)
        )
        oka = oka.to(torch.int64).cumprod(1).bool()
        ha, fa = hot.indices, cold.indices
        sa = st.gather(1, fa)
        home = oka & (sa == -2)
        isg = oka & (sa >= 0)
        vhp = torch.where(
            sa == -2, home_hp[fa], torch.where(sa >= 0, sa, torch.full_like(sa, H))
        )  # victim's host page (H: a hole)
        gsc = sc.gather(1, fa)
        # a home page in place: the group reads it from host from now on; a guest: back to host
        self.gredir[rows, sinkT(home, fa)] = (D + vhp).to(torch.int32)
        self.gredir[rows, sinkT(isg, D + sa)] = -1
        self.ghost[rows, sinkH(isg, sa)] = True
        self.gscore[rows, sinkT(oka & (vhp < H), D + vhp)] = gsc
        self.gfr[rows, sinkD(oka, fa)] = -3
        self.ghost[rows, sinkH(oka, ha)] = False
        self.ghost[:, H] = False
        self.gfr[:, D] = -2
        fr = torch.cat((fb, fa), 1)
        hp = torch.cat((home_hp[fb], ha), 1)
        ok = torch.cat((okb, oka), 1)
        back = torch.cat((torch.ones_like(okb), torch.zeros_like(oka)), 1)
        src = (self.host_row0 + hp * P).reshape(-1).contiguous()
        dst = (fr * P).reshape(-1).contiguous()
        okf = ok.reshape(-1).to(torch.int32).contiguous()
        n = int(okf.numel())
        if self._ggrp is None or self._ggrp.numel() != n:
            self._ggrp = (
                torch.arange(st.shape[0], device=st.device)
                .repeat_interleave(2 * m)
                .contiguous()
            )
        cur = device_module.current_stream()
        row = int(coordinator.mem_pool_device.bytes_per_token)
        tile = 4096
        with device_module.stream(self._stream):
            if coordinator.decode_producer_stream is not None:
                self._stream.wait_stream(coordinator.decode_producer_stream)
            self._stream.wait_stream(cur)
            paging_kernels.copy_group_pages[
                (n, self._lmax, (P * row + tile - 1) // tile)
            ](
                self.gptrs,
                self._ggrp,
                src,
                dst,
                okf,
                LMAX=self._lmax,
                PAGE_BYTES=P * row,
                ROW=row,
                TILE=tile,
            )
            ev = device_module.Event()
            ev.record(self._stream)
        self._gpending = (
            ev,
            fr,
            hp,
            back,
            ok,
            src,
            dst,
            okf,
        )  # the index tensors stay referenced until the copy ran
        cnt = ok.sum()
        self._gmoved += cnt
        self._gmoved_pin.copy_(cnt.reshape(1), non_blocking=True)
        self.gstats[0] += 1
        if self.gstats[0] % 200 == 1:
            self._log()

    def _log(self) -> None:
        coordinator = self._coordinator
        dev, hst = (int(x) for x in self._sel.tolist())  # one sync per log line
        self.gstats[1] = int(self._gmoved)
        d0, h0 = self._sel_logged
        self._sel_logged = (dev, hst)
        tot, rec = dev + hst, (dev - d0) + (hst - h0)
        logger.info(
            "HiSparse paged: step %d; %d evictions, %d pages evicted, %d seals (%d tokens), %d failed reclaims; "
            "%d pages on host, %d evictable in HBM, %d resident "
            "requests; selected rows read from host: %.2f%% since the last line, %.2f%% overall (%d rows); "
            "group moves %s",
            self._steps,
            *self.stats,
            self._n_onhost,
            self.num_evictable_pages,
            len(coordinator._resident),
            100.0 * (hst - h0) / max(rec, 1),
            100.0 * hst / max(tot, 1),
            tot,
            self.gstats,
        )
        if self.exclusive:
            logger.info(
                "HiSparse paged exclusive: %d pages in HBM without a host slot, %d with one, %d on host; "
                "host slots free %d of %d; [pages sealed for eviction, host slots given up, of those after a "
                "copy back, pages promoted, frames moved in band take-backs, staging copies skipped, frames moved "
                "up into the band] %s",
                self.num_exclusive_pages,
                self.num_evictable_pages,
                self._n_onhost,
                coordinator.mem_pool_host.available_size(),
                coordinator.mem_pool_host.size,
                self.xstats,
            )
        if self._rc_rows:
            a = [int(v) for v in self.rc_stat.tolist()]
            b = self._rc_logged
            self._rc_logged = tuple(a)
            d = [a[i] - b[i] for i in range(3)]
            logger.info(
                "HiSparse paged row cache: selected host rows since the last line: %.1f%% read from the cache, %.1f%% "
                "installed (read in place by the indexer layer), %.1f%% in place; totals %s",
                100.0 * d[0] / max(sum(d), 1),
                100.0 * d[1] / max(sum(d), 1),
                100.0 * d[2] / max(sum(d), 1),
                a,
            )

    def evict(self, n_pages: int, lo_frame: int = 0, hi_frame: int = -1) -> int:
        """The n coldest evictable pages leave HBM: their map entries now name their host rows, their frames are free.
        No copy. The caller ordered this after the in-flight batch."""
        coordinator = self._coordinator
        k = min(int(n_pages), self.num_evictable_pages)
        if k <= 0:
            return 0
        P = coordinator.page_size
        self._gfinish(block=True)  # no group move in flight while frames change hands
        self._fold()
        cand = self.evictable
        if lo_frame > 0 or hi_frame >= 0:  # only frames in [lo_frame, hi_frame)
            fr = torch.arange(cand.numel(), device=cand.device)
            cand = cand & (fr >= lo_frame)
            if hi_frame >= 0:
                cand = cand & (fr < hi_frame)
            k = min(k, int(cand.sum()))
            if k <= 0:
                return 0
        score = torch.where(cand, self.score, torch.full_like(self.score, float("inf")))
        sel = torch.topk(score, k, largest=False).indices
        lp = self.frame_lp[sel]
        ar = self._arange
        mapping = (
            coordinator.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
        )
        mapping[(lp[:, None] * P + ar[None, :]).reshape(-1)] = (
            self.host_row0 + self.host_first[lp][:, None] + ar[None, :]
        ).reshape(-1)
        hp = self.host_first[lp] // P
        self._gdrop(sel, hp)
        self.evictable[sel] = False
        self.frame_lp[sel] = -1
        coordinator.token_to_kv_pool_allocator.free_hisparse_indices(
            (sel[:, None] * P + ar[None, :]).reshape(-1)
        )
        self.num_evictable_pages -= k
        self._n_onhost += k
        coordinator._hs_pt_memo = None
        self.stats[0] += 1
        self.stats[1] += k
        if self.stats[0] % 20 == 1:
            self._log()
        return k

    def reclaim(self, need_slots: int) -> bool:
        """Free device slots by evicting cold pages (sealing decode pages first when the sealed ones do not cover it)."""
        coordinator = self._coordinator
        dev = coordinator.token_to_kv_pool_allocator.hisparse_attn_allocator
        short = need_slots - dev.available_size()
        if short <= 0:
            return True
        # Ordered after the in-flight batch (it may still read or write these pages) and after the pending host backup.
        if coordinator.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(
                coordinator.decode_producer_stream
            )
        coordinator.wait_for_pending_backup()
        need_pages = -(-short // coordinator.page_size)
        if self.exclusive:
            self._seal_coldest(max(need_pages, self.evict_batch_pages))
        elif self.num_evictable_pages < need_pages:
            self.seal_all()
        self.evict(max(need_pages, self.evict_batch_pages))
        ok = dev.available_size() >= need_slots
        if not ok:
            self.stats[4] += 1
        return ok

    def release(self, rid: int) -> None:
        """A resident request ends: its pages are no longer eviction candidates (its frames are freed by the allocator,
        through the map; pages on host hold no frame)."""
        coordinator = self._coordinator
        end = self.registered_len.pop(rid, 0)
        if end <= 0:
            return
        P = coordinator.page_size
        mapping = (
            coordinator.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
        )
        first = coordinator.req_to_token_pool.req_to_token[rid, 0:end:P].to(torch.int64)
        dev_first = mapping[first]
        on_dev = dev_first < self.host_row0
        frames = dev_first[on_dev] // P
        num_exclusive = 0
        if self.exclusive:
            self.host_len.pop(rid, None)
            num_exclusive = int(self.xframe[frames].sum())
            self.xframe[frames] = False
            self.num_exclusive_pages -= num_exclusive
            # only the pages that hold a host slot have group state on host pages
            first = first[self.host_first[first // P] >= 0]
        self.evictable[frames] = False
        self.frame_lp[frames] = -1
        hp = self.host_first[first // P] // P
        self._gfinish(block=True)
        self._gdrop(
            frames, None
        )  # the request's home frames: guests back to host, redirects cleared
        D, H = self.num_device_pages, self._host_pages
        r = self.gredir[:, D + hp].to(
            torch.int64
        )  # [G, pages]: frames where the request's pages are guests
        isg = r >= 0
        self.gfr[self._grows, torch.where(isg, r, D)] = -1  # they become holes
        self.gscore[self._grows, torch.where(isg, r, D + H)] = 0
        self.gredir[:, D + hp] = -1
        self.ghost[:, hp] = False
        self.gscore[:, D + hp] = 0
        self.gfr[:, D] = -2
        if self._rc_rows:
            self.rc_epoch[hp] += 1
        self.host_first[first // P] = -1
        n_dev = int(on_dev.sum())
        self.num_evictable_pages -= n_dev - num_exclusive
        self._n_onhost -= end // P - n_dev
