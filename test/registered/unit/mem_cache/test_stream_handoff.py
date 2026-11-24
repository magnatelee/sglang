"""A tensor that one stream leaves for another: the reader runs after the maker, and the memory stays.

Lifetime: PyTorch hands a dropped tensor's memory back to the stream that made it. A reader queued on another
stream that has not run yet (it is fenced behind the batch in flight) then reads what was written there since:
a free list got page ids of 1e9 that way (the merge), then page ids of 263172 = 0x01010101 // 64 (a kept
scratch block overwritten by a mask of ones). Order: a reader on a stream that does not follow the maker's can
run first: with the pre-allocation moved behind the launch, an allocation read a free list whose merge had not
run and handed out slots outside the pool. The maker marks such a tensor with `staged`, every reader takes it
through `handed_over`.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers import hisparse_coordinator as hc
from sglang.srt.managers import hisparse_paging as hp
from sglang.srt.mem_cache import stream_handoff
from sglang.srt.mem_cache.allocator import hisparse as hisparse_allocator
from sglang.srt.mem_cache.allocator import paged as paged_allocator
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 4


class _Event:
    def __init__(self):
        self.done = False

    def record(self, stream):
        self.stream = stream

    def query(self):
        return self.done


class _Stream:
    def __init__(self, name):
        self.name = name
        self.waits = []

    def wait_event(self, event):
        self.waits.append(event)


class _Device:
    """The stream that is current, and events the test completes."""

    def __init__(self, stream):
        self.current = stream

    def current_stream(self):
        return self.current

    def Event(self):
        return _Event()


def _tensor():
    return SimpleNamespace(is_cuda=True, record_stream=MagicMock())


class _Taken:
    """Stands in for `handed_over` in a reader's module: which tensors the reader took through it."""

    def __init__(self):
        self.tensors = []

    def __call__(self, tensor):
        self.tensors.append(tensor)
        return tensor

    def took(self, tensor) -> bool:
        return any(t is tensor for t in self.tensors)


def test_a_device_tensor_is_held_for_the_stream_that_reads_it():
    tensor = _tensor()
    with patch.object(
        stream_handoff.device_module, "current_stream", return_value="reader"
    ):
        assert stream_handoff.handed_over(tensor) is tensor
        stream_handoff.handed_over(tensor)
    tensor.record_stream.assert_called_once_with("reader")  # once per reading stream


def test_a_reader_waits_on_the_device_for_a_maker_that_has_not_run():
    maker, reader = _Stream("maker"), _Stream("reader")
    tensor = _tensor()
    with patch.object(stream_handoff, "device_module", _Device(maker)):
        assert stream_handoff.staged(tensor) is tensor
    made = tensor.__dict__[stream_handoff._MADE]
    assert made.stream is maker

    with patch.object(stream_handoff, "device_module", _Device(reader)):
        stream_handoff.handed_over(tensor)
    assert reader.waits == [made]  # a wait queued on the reader's stream; the host waited for nothing
    tensor.record_stream.assert_called_once_with(reader)


def test_a_maker_that_has_run_is_not_waited_for_again():
    maker, reader = _Stream("maker"), _Stream("reader")
    tensor = _tensor()
    with patch.object(stream_handoff, "device_module", _Device(maker)):
        stream_handoff.staged(tensor)
    tensor.__dict__[stream_handoff._MADE].done = True

    with patch.object(stream_handoff, "device_module", _Device(reader)):
        stream_handoff.handed_over(tensor)
        stream_handoff.handed_over(tensor)
    assert reader.waits == []
    assert stream_handoff._MADE not in tensor.__dict__


def test_nothing_staged_is_nothing_to_stage():
    assert stream_handoff.staged(None) is None
    assert stream_handoff.handed_over(None) is None
    host = torch.arange(4)
    assert stream_handoff.staged(host) is host
    assert stream_handoff._MADE not in host.__dict__


def test_a_host_tensor_has_no_stream_to_hold_it_for():
    tensor = torch.arange(4)
    with patch.object(torch.Tensor, "record_stream") as record_stream:
        assert stream_handoff.handed_over(tensor) is tensor
    record_stream.assert_not_called()


def test_the_merge_takes_the_free_list_and_every_staged_piece():
    """Pieces are staged by whatever stream freed them (a release on its own stream, an eviction under the
    forward stream); the merge marks the list it leaves."""
    allocator = PagedTokenToKVPoolAllocator(
        8 * PAGE, PAGE, torch.bfloat16, "cpu", None, True
    )
    free_before = allocator.free_pages = torch.tensor([5, 6])
    first, second = torch.tensor([8, 2]), torch.tensor([4])
    marked = _Taken()
    with patch.object(paged_allocator, "staged", marked):
        allocator.free_page_ids(first)
        allocator.free_page_ids(second)
    assert marked.took(first) and marked.took(second)
    taken, marked = _Taken(), _Taken()

    with patch.object(paged_allocator, "handed_over", taken), patch.object(
        paged_allocator, "staged", marked
    ):
        allocator.merge_and_sort_free()
        merged = allocator._free_pages

    assert all(taken.took(t) for t in (free_before, first, second))
    assert marked.took(merged)
    assert allocator.free_pages.tolist() == [2, 4, 5, 6, 8]


def test_every_allocation_takes_the_free_list_and_marks_what_it_leaves():
    """The list is merged or sliced on one stream and read by the next allocation on another (the schedule
    stream, the forward stream's verify preparation)."""
    allocator = PagedTokenToKVPoolAllocator(
        8 * PAGE, PAGE, torch.bfloat16, "cpu", None, True
    )
    allocator.use_triton_kernels = False  # host tensors
    for allocate in (
        lambda: allocator.alloc(2 * PAGE),
        lambda: allocator.alloc_extend(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([PAGE]),
            torch.tensor([PAGE]),
            torch.tensor([-1]),
            PAGE,
        ),
        lambda: allocator.alloc_decode(
            torch.tensor([PAGE + 1]), torch.tensor([PAGE + 1]), torch.tensor([PAGE - 1])
        ),
    ):
        before = allocator._free_pages
        taken, marked = _Taken(), _Taken()
        with patch.object(paged_allocator, "handed_over", taken), patch.object(
            paged_allocator, "staged", marked
        ):
            assert allocate() is not None
        assert taken.took(before)
        assert marked.took(allocator._free_pages)
        assert allocator._free_pages is not before


def _spec_swap(blocks, **allocator):
    swap = hc.HiSparseSpecSwapManager.__new__(hc.HiSparseSpecSwapManager)
    swap.enabled = True
    swap._spare_scratch = list(blocks)
    swap._scratch_reqs = set()
    swap.req_to_scratch = torch.zeros((2, 3, 2 * PAGE), dtype=torch.int32)
    swap._coordinator = SimpleNamespace(
        page_size=PAGE, token_to_kv_pool_allocator=SimpleNamespace(**allocator)
    )
    return swap


def test_a_kept_scratch_block_is_marked_where_the_release_keeps_it():
    swap = _spec_swap([])
    swap._coordinator.enable_resident = True
    swap.req_to_scratch[:, 1] = torch.arange(3 * PAGE, 5 * PAGE, dtype=torch.int32)
    swap._scratch_reqs = {1}
    marked = _Taken()

    with patch.object(hc, "staged", marked):
        swap.free_unrotated_scratch(1)

    (block,) = swap._spare_scratch
    assert marked.took(block)
    assert block.tolist() == list(range(3 * PAGE, 5 * PAGE))


def test_held_buffer_slots_are_marked_where_the_pre_allocation_takes_them():
    coordinator = object.__new__(hc.HiSparseCoordinator)
    slots = torch.arange(2 * PAGE, 4 * PAGE)
    coordinator.token_to_kv_pool_allocator = SimpleNamespace(
        alloc_buffer_extra=lambda prefill_tokens: slots
    )
    req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=3))
    marked = _Taken()

    with patch.object(hc, "staged", marked):
        assert coordinator.hold_buffer_slots(req, 100)

    assert marked.took(slots)
    assert coordinator._recv_buffer_hold[3] is slots


def test_a_kept_scratch_block_is_taken_before_its_pages_are_freed():
    """A release keeps the block on its own stream; the page ids are computed when the freeing stream
    gets to them, after the block was dropped."""
    block = torch.arange(3 * PAGE, 5 * PAGE, dtype=torch.int32)
    freed = []
    swap = _spec_swap([block], free_device_pages=lambda pages: freed.append(pages))
    taken = _Taken()

    with patch.object(hc, "handed_over", taken):
        assert swap.free_spare_scratch()

    assert taken.took(block)
    assert [pages.tolist() for pages in freed] == [[3, 4]]
    assert not swap._spare_scratch


def test_a_kept_scratch_block_is_taken_before_an_arrival_copies_it():
    block = torch.arange(3 * PAGE, 5 * PAGE, dtype=torch.int32)
    swap = _spec_swap([block])
    taken = _Taken()

    with patch.object(hc, "handed_over", taken):
        swap.allocate_scratch(1)

    assert taken.took(block)
    assert swap.req_to_scratch[:, 1].tolist() == [block.tolist()] * 2
    assert swap._scratch_reqs == {1}


def test_held_buffer_slots_are_taken_before_their_pages_are_freed():
    """Taken at pre-allocation on the schedule stream, given back by a release on its own."""
    allocator = HiSparseTokenToKVPoolAllocator.__new__(HiSparseTokenToKVPoolAllocator)
    allocator.page_size = PAGE
    allocator._hs_el = None
    freed = []
    allocator.hisparse_attn_allocator = SimpleNamespace(
        free_group=None, free_page_ids=lambda pages: freed.append(pages)
    )
    slots = torch.arange(2 * PAGE, 4 * PAGE)
    taken = _Taken()

    with patch.object(hisparse_allocator, "handed_over", taken):
        allocator.free_buffer_extra(slots)

    assert taken.took(slots)
    assert [pages.tolist() for pages in freed] == [[2, 3]]


def test_a_group_move_s_plan_is_taken_by_whoever_completes_it():
    """Planned on the stream that started the move; a release completes it on its own."""
    paging = object.__new__(hp.HiSparsePagingManager)
    groups, frames, host_pages = 2, 3, 2
    paging.num_device_pages, paging._host_pages = frames, host_pages
    paging._grows = torch.arange(groups)[:, None]
    total = frames + host_pages
    paging.gredir = torch.full((groups, total + 1), -1, dtype=torch.int32)
    paging.gfr = torch.full((groups, frames + 1), -2, dtype=torch.int32)
    paging.gscore = torch.zeros((groups, total + 1))
    done = SimpleNamespace(query=lambda: True)
    plan = tuple(
        torch.zeros((groups, 1), dtype=dtype)
        for dtype in (torch.int64, torch.int64, torch.bool, torch.bool)
    )
    paging._gpending = (done, *plan)
    taken = _Taken()

    with patch.object(hp, "handed_over", taken):
        assert paging._gfinish(block=False)

    assert all(taken.took(t) for t in plan)
    assert paging._gpending is None
