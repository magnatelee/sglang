"""The device pages a paged HiSparse allocator hands out stop at the band's last page."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 64
ROW = 576  # a raw MLA row; a 2 MiB chunk holds 56.9 pages of such rows
CHUNK = 2 << 20


def _make_allocator(*, size: int, band_max: int, mapped: int):
    allocator = HiSparseTokenToKVPoolAllocator.__new__(HiSparseTokenToKVPoolAllocator)
    allocator.page_size = PAGE
    allocator.device = "cpu"
    allocator._hs_el = dict(
        kv=SimpleNamespace(mapped=mapped),
        row=ROW,
        k_min=size,
        k_max=size + band_max,
        dev_max_page=size // PAGE,
        log_max_page=0,
    )
    allocator.hisparse_attn_allocator = SimpleNamespace(
        free_pages=torch.arange(1, size // PAGE + 1),
        staged_pages=[],
        merge_and_sort_free=lambda: None,
    )
    allocator.logical_attn_allocator = SimpleNamespace(
        free_pages=torch.empty(0, dtype=torch.int64),
        staged_pages=[],
        merge_and_sort_free=lambda: None,
    )
    return allocator


def _chunks(rows: int) -> int:
    return -(-rows * ROW // CHUNK) * CHUNK


def test_the_last_chunk_s_tail_is_not_handed_out():
    """The mapped bytes are whole chunks, so the last chunk can hold whole pages
    of rows past the band's last page; the paging tables have no entry for them."""
    size, band_max = 65536, 25024
    last_page = (size + band_max) // PAGE
    mapped = _chunks(size + band_max + PAGE)
    assert mapped // ROW // PAGE - 1 > last_page  # the tail holds whole pages
    allocator = _make_allocator(size=size, band_max=band_max, mapped=mapped)

    assert allocator._hs_el_mapped_rows() == size + band_max + PAGE
    allocator.hs_el_set_limit("dev", mapped // ROW // PAGE - 1)

    assert allocator._hs_el["dev_max_page"] == last_page
    assert int(allocator.hisparse_attn_allocator.free_pages.max()) == last_page


def test_a_partly_grown_band_keeps_its_mapped_limit():
    size, band_max = 65536, 25024
    mapped = _chunks(size + PAGE) + 3 * CHUNK
    allocator = _make_allocator(size=size, band_max=band_max, mapped=mapped)

    assert allocator._hs_el_mapped_rows() == mapped // ROW
    allocator.hs_el_set_limit("dev", mapped // ROW // PAGE - 1)

    assert allocator._hs_el["dev_max_page"] == mapped // ROW // PAGE - 1
    assert allocator._hs_el["dev_max_page"] < (size + band_max) // PAGE


def test_a_pool_without_a_band_stops_at_its_own_last_page():
    """Ranges pinned for a PD prefill have no band: the alignment slack of the
    static range is such a tail from the start."""
    size = 149952
    mapped = _chunks(size + PAGE)
    assert mapped // ROW // PAGE - 1 > size // PAGE
    allocator = _make_allocator(size=size, band_max=0, mapped=mapped)

    assert allocator._hs_el_mapped_rows() // PAGE - 1 == size // PAGE


def _with_limit_wrappers(*, pinned: bool):
    size = 65536
    allocator = _make_allocator(
        size=size, band_max=0 if pinned else 25024, mapped=_chunks(size + PAGE)
    )
    allocator._hs_el.update(
        pinned=pinned, ik=SimpleNamespace(mapped=1 << 30), ik_page_bytes=1
    )
    released = []
    for inner in (allocator.hisparse_attn_allocator, allocator.logical_attn_allocator):
        inner._release_page_ids = lambda *page_ids: released.extend(page_ids)
        inner.num_pages = size // PAGE
    allocator._hs_el_init_limits()
    return allocator, released


def test_pinned_ranges_free_and_merge_without_a_mask():
    """The limits of pinned ranges never move, so no freed page lies above them:
    a mask would only read back, which waits for the batch in flight."""
    allocator, released = _with_limit_wrappers(pinned=True)
    device = allocator.hisparse_attn_allocator
    pages = torch.tensor([5, 7])

    device._release_page_ids(pages)
    assert released[-1] is pages

    free_pages = device.free_pages
    device.merge_and_sort_free()
    assert device.free_pages is free_pages


def test_a_lent_band_keeps_pages_above_its_limit_out():
    allocator, released = _with_limit_wrappers(pinned=False)
    device = allocator.hisparse_attn_allocator
    limit = allocator._hs_el["dev_max_page"]

    device._release_page_ids(torch.tensor([5, limit + 3]))
    assert released[-1].tolist() == [5]

    device.free_pages = torch.tensor([2, limit + 1])
    device.merge_and_sort_free()
    assert device.free_pages.tolist() == [2]


def test_pinned_ranges_take_pages_below_the_band_without_a_mask():
    """Scratch and buffers come from below the band; pinned ranges have none,
    so the lowest free pages are taken as they are, as the masked path would."""
    size = 65536
    taken = {}
    for pinned in (True, False):
        allocator = _make_allocator(size=size, band_max=0, mapped=_chunks(size + PAGE))
        allocator._hs_el.update(pinned=pinned)
        device = allocator.hisparse_attn_allocator
        device.free_pages = torch.arange(3, size // PAGE + 1)

        slots = allocator.hs_alloc_below(2 * PAGE, size + PAGE)

        taken[pinned] = (slots.tolist(), device.free_pages.tolist())
        assert slots[::PAGE].tolist() == [3 * PAGE, 4 * PAGE]
        assert allocator.hs_alloc_below(size, size + PAGE) is None
    assert taken[True] == taken[False]

    pinned = _make_allocator(size=size, band_max=0, mapped=_chunks(size + PAGE))
    pinned._hs_el.update(pinned=True)
    buffer_slots = torch.arange(PAGE, 2 * PAGE)
    assert pinned._hs_el_buffer_below(buffer_slots) is buffer_slots


def test_pinned_merge_reads_nothing_back_and_leaves_the_staged_pages_to_the_merge():
    """The pinned merge has no mask: nothing waits for it. Keeping its inputs' memory until it has
    run is the merge's own job (test_stream_handoff)."""
    size = 65536
    allocator = _make_allocator(size=size, band_max=0, mapped=_chunks(size + PAGE))
    allocator._hs_el.update(
        pinned=True, ik=SimpleNamespace(mapped=1 << 30), ik_page_bytes=1
    )
    device = allocator.hisparse_attn_allocator
    merges = []
    device.merge_and_sort_free = lambda: merges.append(list(device.staged_pages))
    for inner in (device, allocator.logical_attn_allocator):
        inner._release_page_ids = lambda *page_ids: None
        inner.num_pages = size // PAGE
    allocator._hs_el_init_limits()
    merges.clear()
    staged = torch.tensor([7, 3])
    device.staged_pages = [staged]
    free_before = device.free_pages

    device.merge_and_sort_free()

    assert len(merges) == 1 and merges[0][0] is staged
    assert device.free_pages is free_before
