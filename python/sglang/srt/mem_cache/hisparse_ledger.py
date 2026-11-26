"""Admission numbers for paged HiSparse under a split context (--dcp-size).

The GPUs of a split-context group run one scheduler each and must take every decision
alike, or their model collectives stop matching. Which pages sit in a device frame and
which on host differs between them (each pages its own share), and so do their free
frames at any moment: copies to host and group moves complete at different times. So
nothing here reads a device or host free list, or a counter that such work moves. The
ledger counts pages from what every GPU knows identically (the logical allocator sees
the same calls on all of them: under a split no release is deferred):

  live     pages held by requests
  movable  the whole pages below the writing position of resident requests: they may
           leave the device pool (they have a host copy or can get one)
  pinned   live - movable: pages being prefilled, staged or written

A request that needs n pages fits when
  n <= logical pages not live,  pinned + n <= D  and  live + n <= D + H
(D device frames, H host pages). The first is the id space, the second says the new
pages can be written, the third that every page has a place. A GPU realizes what was
admitted by waiting for its own pending work (seals, evictions).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

# Digests are kept below 2**62 so that their negation fits the int64 the audit reduces.
_DIGEST_MASK = (1 << 62) - 1


class HiSparsePageLedger:
    def __init__(self, coordinator: HiSparseCoordinator) -> None:
        self._coordinator = coordinator
        allocator = coordinator.token_to_kv_pool_allocator
        self._allocator = allocator
        self._rows_per_page = allocator.row_page_size
        self._tokens_per_page = allocator.page_size
        logical = allocator.logical_attn_allocator
        self.logical_pages = logical.available_size() // self._tokens_per_page
        # Frames requests can use: everything free in the empty pool (the padding page
        # and rows taken for the server's lifetime are not).
        self.device_pages = (
            allocator.hisparse_attn_allocator.available_size() // self._rows_per_page
        )
        self.host_pages = coordinator.mem_pool_host.available_size() // self._rows_per_page

    def counts(self) -> tuple[int, int]:
        """(live, movable) pages, from the requests' lengths and states alone."""
        coordinator = self._coordinator
        logical = self._allocator.logical_attn_allocator
        live = self.logical_pages - logical.available_size() // self._tokens_per_page
        sealable_end = coordinator.paging.sealable_end
        movable = sum(
            sealable_end(req) // self._rows_per_page
            for req in coordinator._resident.values()
        )
        return live, movable

    def available_pages(self) -> int:
        live, movable = self.counts()
        pinned = live - movable
        return max(
            min(
                self.logical_pages - live,
                self.device_pages - pinned,
                self.device_pages + self.host_pages - live,
            ),
            0,
        )

    def available_tokens(self) -> int:
        return self.available_pages() * self._tokens_per_page

    def holds_request_tokens(self) -> bool:
        """Whether a request holds logical slots."""
        return self.counts()[0] > 0

    def digest(self) -> int:
        """What the GPUs of a group must agree on: the constants and the counts, of which
        every number handed to a scheduler is a function. (How often a scheduler asks is
        its own business: rank 0 also asks for its metrics.)"""
        live, movable = self.counts()
        value = hash(
            (self.logical_pages, self.device_pages, self.host_pages, live, movable)
        )
        return value & _DIGEST_MASK
