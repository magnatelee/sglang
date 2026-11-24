import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers import hisparse_paging as hp
from sglang.srt.managers.hisparse_paging import HiSparsePagingManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 4
FRAMES = 6


class _Stream:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def wait_stream(self, stream):
        self.log.append((self.name, "waits for", stream.name))


class _Event:
    """A copy's completion: the test decides when it has run."""

    def __init__(self):
        self.done = False
        self.waited = False

    def record(self, stream):
        self.stream = stream.name

    def query(self):
        return self.done

    def synchronize(self):
        self.waited = True
        self.done = True


class _DeviceModule:
    """The stream that is current, and the order the streams are put in."""

    def __init__(self, log):
        self.current = _Stream("scheduler", log)
        self.events = []

    def current_stream(self):
        return self.current

    @contextlib.contextmanager
    def stream(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous

    def Event(self):
        self.events.append(_Event())
        return self.events[-1]


class _Host:
    def __init__(self, free_slots, device):
        self.free_slots = free_slots
        self.device = device
        self.copies = []
        self.freed = []
        self.next_slot = 400

    def available_size(self):
        return self.free_slots

    def alloc(self, need):
        self.free_slots -= need
        first, self.next_slot = self.next_slot, self.next_slot + need
        return torch.arange(first, first + need)

    def free(self, slots):
        self.free_slots += slots.numel()
        self.freed.extend(slots.tolist())

    def backup_from_device_all_layer(self, pool, host_locs, device_locs, io_backend):
        self.copies.append(
            (host_locs.tolist(), device_locs.tolist(), self.device.current.name)
        )


class TestBackgroundSealOfPromptPages(CustomTestCase):
    """Exclusive placement under the overlap loop: an arrival that needs frames
    cold pages still occupy starts their copy to host beside the running steps
    and waits; nothing on the scheduler's stream waits for that copy."""

    def setUp(self):
        super().setUp()
        self.log = []
        self.device = _DeviceModule(self.log)
        patcher = patch.object(hp, "device_module", self.device)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.host = _Host(64 * PAGE, self.device)
        self.free_frames = [0]

    def _paging(self):
        paging = object.__new__(HiSparsePagingManager)
        paging._coordinator = SimpleNamespace(
            page_size=PAGE,
            device="cpu",
            mem_pool_host=self.host,
            mem_pool_device=object(),
            decode_producer_stream=_Stream("forward", self.log),
            _release_stream=_Stream("release", self.log),
            req_to_host_pool=torch.full((2, 16 * PAGE), -1, dtype=torch.int64),
            token_to_kv_pool_allocator=SimpleNamespace(
                hisparse_attn_allocator=SimpleNamespace(
                    available_size=lambda: self.free_frames[0] * PAGE
                )
            ),
            _resident={},
        )
        paging.exclusive = True
        paging.num_device_pages = FRAMES
        paging.evict_batch_pages = 1
        paging._arange = torch.arange(PAGE)
        # Frames 0-3 hold exclusive pages, coldest first; frame 4 an evictable one.
        paging.xframe = torch.tensor([True, True, True, True, False, False])
        paging.evictable = torch.tensor([False, False, False, False, True, False])
        paging.score = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        paging.frame_lp = torch.tensor([10, 11, 12, 13, 14, -1])
        paging.lp_prompt = torch.zeros(16, dtype=torch.bool)
        paging.lp_prompt[torch.tensor([10, 12, 13])] = True  # page 11: written by decode
        paging.lp_rid = torch.zeros(16, dtype=torch.int64)
        paging.lp_pos = torch.arange(16)
        paging.host_first = torch.full((16,), -1, dtype=torch.int64)
        paging.gscore = torch.ones(2, FRAMES + 512)
        paging.num_exclusive_pages = 4
        paging.num_evictable_pages = 1
        paging.registered_len = {}
        paging._gpending = None
        paging._check = False
        paging._pin = False
        paging._seal_stream = _Stream("seal", self.log)
        paging._seal_pending = None
        paging._seal_serial = 0
        paging._seal_unlisted = {}
        paging._seal_dry = False
        paging._seal_ok_pin = torch.empty(FRAMES, dtype=torch.bool)
        paging.early_seals = [0, 0]
        paging.stats = [0, 0, 0, 0, 0]
        paging.xstats = [0, 0, 0, 0, 0, 0, 0]
        return paging

    def test_copy_runs_on_its_own_stream_and_pages_stay_exclusive_until_it_ends(self):
        paging = self._paging()
        self.assertTrue(paging._seal_prompt_pages(2))
        # The two coldest prompt pages; frame 1 is a decode page.
        (host_locs, device_locs, stream), = self.host.copies
        self.assertEqual(stream, "seal")
        self.assertEqual(device_locs, list(range(0, PAGE)) + list(range(8, 12)))
        self.assertEqual(self.log, [("seal", "waits for", "scheduler")])
        # Nothing took effect: an eviction cannot pick a page whose copy is running.
        self.assertFalse(paging.finish_seal())
        self.assertEqual(paging.xframe.tolist(), [True, True, True, True, False, False])
        self.assertEqual((paging.num_exclusive_pages, paging.num_evictable_pages), (4, 1))

        self.device.events[-1].done = True
        self.assertTrue(paging.finish_seal())
        self.assertFalse(self.device.events[-1].waited)
        self.assertEqual(paging.xframe.tolist(), [False, True, False, True, False, False])
        self.assertEqual(
            paging.evictable.tolist(), [True, False, True, False, True, False]
        )
        self.assertEqual((paging.num_exclusive_pages, paging.num_evictable_pages), (2, 3))
        self.assertEqual(paging.host_first[torch.tensor([10, 12])].tolist(), [400, 404])
        self.assertEqual(paging.early_seals, [1, 2])
        # The seal took effect on the release stream; nothing waited for the copy.
        self.assertEqual(
            self.log,
            [("seal", "waits for", "scheduler"), ("scheduler", "waits for", "release")],
        )

    def test_the_picks_are_taken_by_the_stream_the_seal_takes_effect_on(self):
        """Picked on the stream that started the seal, read on the release stream."""
        paging = self._paging()
        self.assertTrue(paging._seal_prompt_pages(2))
        picked = paging._seal_pending[1:3]
        taken = []

        def handed_over(tensor):
            taken.append((tensor, self.device.current.name))
            return tensor

        self.device.events[-1].done = True
        with patch.object(hp, "handed_over", handed_over):
            self.assertTrue(paging.finish_seal())

        for pick in picked:
            self.assertTrue(
                any(t is pick and stream == "release" for t, stream in taken)
            )

    def test_the_picks_are_marked_on_the_stream_that_picks_them(self):
        paging = self._paging()
        marked = []
        with patch.object(
            hp, "staged", lambda t: marked.append((t, self.device.current.name)) or t
        ):
            self.assertTrue(paging._seal_prompt_pages(2))
        sel, lp = paging._seal_pending[1:3]
        for pick in (sel, lp):
            self.assertTrue(any(t is pick and stream == "scheduler" for t, stream in marked))

    def test_one_copy_at_a_time_and_only_into_free_host_slots(self):
        paging = self._paging()
        self.host.free_slots = PAGE
        self.assertTrue(paging._seal_prompt_pages(3))
        self.assertEqual(len(self.host.copies[0][1]), PAGE)  # one page: frame 0
        self.assertFalse(paging._seal_prompt_pages(3))  # one is running
        self.device.events[-1].done = True
        paging.finish_seal()
        self.assertFalse(paging._seal_prompt_pages(3))  # host pool full
        self.assertEqual(len(self.host.copies), 1)

    def test_picks_that_are_not_prompt_pages_give_their_host_slots_back(self):
        paging = self._paging()
        self.assertTrue(paging._seal_prompt_pages(4))  # three such pages exist
        self.device.events[-1].done = True
        paging.finish_seal()
        self.assertEqual(paging.xframe.tolist(), [False, True, False, False, False, False])
        self.assertEqual((paging.num_exclusive_pages, paging.num_evictable_pages), (1, 4))
        self.assertEqual(len(self.host.freed), PAGE)
        self.assertEqual(paging.early_seals, [1, 3])
        # Nothing is left to copy beside the steps until a prompt arrives.
        self.assertTrue(paging._seal_dry)

    def test_arrival_gets_frames_only_from_free_ones_and_completed_copies(self):
        paging = self._paging()
        self.free_frames[0] = 1
        self.assertTrue(paging.room_without_copy(1 * PAGE))  # free
        self.assertTrue(paging.room_without_copy(2 * PAGE))  # + the evictable page
        self.assertEqual(self.host.copies, [])
        # Three frames: one more page must leave first. Its copy starts; ask again.
        self.assertFalse(paging.room_without_copy(3 * PAGE))
        self.assertEqual(len(self.host.copies), 1)
        self.assertFalse(paging.room_without_copy(3 * PAGE))
        self.assertEqual(len(self.host.copies), 1)
        self.device.events[-1].done = True
        self.assertTrue(paging.room_without_copy(3 * PAGE))
        self.assertFalse(self.device.events[-1].waited)

    def test_without_prompt_pages_the_ordered_reclaim_decides(self):
        paging = self._paging()
        paging.lp_prompt[:] = False
        paging._seal_dry = True
        # Decode pages can still be sealed on the scheduler's stream.
        self.assertTrue(paging.room_without_copy(4 * PAGE))
        # More than every registered page: wait for requests to end or arrive.
        self.assertFalse(paging.room_without_copy(7 * PAGE))
        self.assertEqual(self.host.copies, [])

    def test_blocking_seal_waits_for_the_running_copy_first(self):
        paging = self._paging()
        paging._seal_prompt_pages(1)
        self.assertTrue(paging.finish_seal(block=True))
        self.assertTrue(self.device.events[-1].waited)
        self.assertEqual(paging.num_exclusive_pages, 3)

    def test_ended_request_leaves_the_candidates(self):
        paging = self._paging()
        paging._coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(40, 40 + 2 * PAGE).repeat(2, 1)
        )
        paging.registered_len = {1: 2 * PAGE}  # logical pages 10 and 11
        paging._seal_prompt_pages(1)
        paging.unlist(1)
        self.assertEqual(paging._seal_unlisted, {1: 1})
        self.assertFalse(bool(paging.lp_prompt[10]))
        self.assertTrue(bool(paging.lp_prompt[12]))


if __name__ == "__main__":
    unittest.main()
