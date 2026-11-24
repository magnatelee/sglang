"""Guards of the paged HiSparse tables: a breach means a running request reads rows that are not its own.

Seen in a deployment: a frame freed while a request still used it was handed to a second request; both were
registered on it, the first owner's flags stayed, and an eviction then took the second owner's page, which had
no host copy. Each guard stops the server at the first such state instead of serving the rows.
"""

import collections
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers import hisparse_paging as hp
from sglang.srt.managers.hisparse_paging import HiSparsePagingManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 4


class _Event:
    def __init__(self):
        self.done = False
        self.synchronized = False

    def record(self, stream):
        self.stream = stream

    def query(self):
        return self.done

    def synchronize(self):
        self.synchronized = True
        self.done = True


class _DeviceModule:
    def __init__(self):
        self.events = []

    def Event(self):
        self.events.append(_Event())
        return self.events[-1]

    def current_stream(self):
        return "scheduler"


def _paging():
    paging = object.__new__(HiSparsePagingManager)
    paging._coordinator = SimpleNamespace(page_size=PAGE)
    paging._guard_counts = torch.zeros(HiSparsePagingManager.GUARD_SLOTS, dtype=torch.int64)
    paging._guard_pending = collections.deque()
    paging._guard_serial = 0
    paging.guard_breaches = {}
    paging._steps = 7
    paging.exclusive = True
    paging.num_evictable_pages = 3
    paging.num_exclusive_pages = 5
    paging._n_onhost = 2
    paging.registered_len = {1: 8 * PAGE}
    return paging


class TestGuardReadsWithoutWaiting(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.device = _DeviceModule()
        patcher = patch.object(hp, "device_module", self.device)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_count_is_read_only_once_its_stream_has_run_it(self):
        paging = _paging()
        paging._guard("an eviction took pages that have no host copy", torch.tensor(2), "eviction of 9 pages")
        self.assertEqual(self.device.events[0].stream, "scheduler")

        paging.check_guards()  # still queued: nothing is read, nothing waits
        self.assertFalse(self.device.events[0].synchronized)

        self.device.events[0].done = True
        with self.assertRaisesRegex(RuntimeError, "took pages that have no host copy: 2 .eviction of 9 pages"):
            paging.check_guards()
        self.assertEqual(paging.guard_breaches, {"an eviction took pages that have no host copy": 1})

    def test_a_zero_count_passes_and_later_guards_are_still_read(self):
        paging = _paging()
        paging._guard("first", torch.tensor(0), "a")
        paging._guard("second", torch.tensor(1), "b")
        for event in self.device.events:
            event.done = True
        with self.assertRaisesRegex(RuntimeError, "second: 1"):
            paging.check_guards()

    def test_guards_are_read_in_the_order_they_were_issued(self):
        paging = _paging()
        paging._guard("first", torch.tensor(0), "a")
        paging._guard("second", torch.tensor(1), "b")
        self.device.events[1].done = True
        paging.check_guards()  # the first has not run: the second waits its turn
        self.assertEqual(len(paging._guard_pending), 2)

    def test_a_full_ring_waits_for_its_oldest_guard(self):
        paging = _paging()
        for _ in range(HiSparsePagingManager.GUARD_SLOTS):
            paging._guard("clean", torch.tensor(0), "")
        for event in self.device.events[1:]:
            event.done = True
        paging._guard("one more", torch.tensor(0), "")
        self.assertTrue(self.device.events[0].synchronized)
        self.assertEqual(len(paging._guard_pending), 1)

    def test_a_manager_built_without_init_has_no_guard(self):
        paging = object.__new__(HiSparsePagingManager)
        paging._guard("anything", torch.tensor(1), "")
        self.assertEqual(self.device.events, [])


class TestGroupTickPlan(CustomTestCase):
    """A background seal takes effect on another stream while a group tick's plan is queued: the tick must not
    take a frame it sees as evictable when its page had no host slot in the read that gave the host page."""

    def setUp(self):
        super().setUp()
        # frame 1: evictable with host slot 40; frame 2: sealed just now (flag seen, slot not yet in the read);
        # frame 3: exclusive; frame 4: already away for group 0 (a guest of host page 7)
        self.frame_lp = torch.tensor([-1, 10, 11, 12, 13])
        self.host_first = torch.full((16,), -1)
        self.host_first[torch.tensor([10, 13])] = torch.tensor([40, 48])
        self.evictable = torch.tensor([False, True, True, False, True])
        self.state = torch.tensor([[-2, -2, -2, -2, 7], [-2, -2, -2, -2, -2]])

    def _plan(self):
        home_hp, has_host = hp.home_host_pages(
            frame_lp=self.frame_lp, host_first=self.host_first, page_size=PAGE, num_host_pages=16
        )
        elig = hp.guest_frames_eligible(
            evictable=self.evictable, has_host=has_host, group_state=self.state
        )
        return home_hp, has_host, elig

    def test_host_page_and_slot_come_from_the_same_read(self):
        home_hp, has_host, _ = self._plan()
        self.assertEqual(home_hp.tolist(), [0, 10, 0, 0, 12])
        self.assertEqual(has_host.tolist(), [False, True, False, False, True])

    def test_a_frame_flagged_evictable_without_a_slot_is_not_taken(self):
        _, _, elig = self._plan()
        self.assertEqual(elig[1].tolist(), [False, True, False, False, True])
        # group 0: frame 4 is away already and may take another guest
        self.assertEqual(elig[0].tolist(), [False, True, False, False, True])
        self.assertFalse(bool(elig[:, 2].any()))

    def test_once_the_slot_is_in_the_read_the_frame_is_eligible(self):
        self.host_first[11] = 44
        home_hp, _, elig = self._plan()
        self.assertEqual(int(home_hp[2]), 11)
        self.assertTrue(bool(elig[:, 2].all()))

    def test_the_guard_counts_a_home_victim_without_a_slot(self):
        _, has_host, _ = self._plan()
        frames = torch.tensor([[1, 2], [2, 4]])
        home = torch.tensor([[True, True], [False, True]])
        self.assertEqual(
            int(hp.home_victims_without_host_slot(home=home, frames=frames, has_host=has_host)), 1
        )
        home = torch.tensor([[True, False], [False, True]])
        self.assertEqual(
            int(hp.home_victims_without_host_slot(home=home, frames=frames, has_host=has_host)), 0
        )


class TestEvictionGuard(CustomTestCase):
    """Frames 1 and 2 are evictable with host slots; frame 3 holds an exclusive page; frame 4 is free."""

    def setUp(self):
        super().setUp()
        self.candidates = torch.tensor([False, True, True, False, False])
        self.frame_lp = torch.tensor([-1, 10, 11, 12, -1])
        self.host_first = torch.full((16,), -1)
        self.host_first[torch.tensor([10, 11])] = torch.tensor([40, 44])

    def _count(self, picks):
        return int(
            hp.picks_without_host_copy(
                picks=torch.tensor(picks),
                candidates=self.candidates,
                frame_lp=self.frame_lp,
                host_first=self.host_first,
            )
        )

    def test_evictable_pages_with_their_host_slots_pass(self):
        self.assertEqual(self._count([1, 2]), 0)

    def test_a_pick_that_fills_up_with_other_frames_is_counted(self):
        # the host's counter said three, two frames are flagged: the pick takes the padding page too
        self.assertEqual(self._count([1, 2, 0]), 1)
        self.assertEqual(self._count([1, 3, 4]), 2)

    def test_a_flagged_frame_whose_page_has_no_host_slot_is_counted(self):
        self.candidates[3] = True  # a second owner registered on a frame that kept its old flag
        self.assertEqual(self._count([1, 3]), 1)


class TestRegistrationGuard(CustomTestCase):
    """Five frames of four rows: frame 1 holds a registered exclusive page, frame 2 a stale evictable flag."""

    def setUp(self):
        super().setUp()
        self.frame_lp = torch.tensor([-1, 10, -1, -1, -1])
        self.evictable = torch.tensor([False, False, True, False, False])
        self.exclusive = torch.tensor([False, True, False, False, False])

    def _count(self, dev_first, exclusive="set"):
        return int(
            hp.frames_not_free_for_registration(
                dev_first=torch.tensor(dev_first),
                page_size=PAGE,
                frame_lp=self.frame_lp,
                evictable=self.evictable,
                exclusive=self.exclusive if exclusive == "set" else None,
            )
        )

    def test_free_frames_pass(self):
        self.assertEqual(self._count([3 * PAGE, 4 * PAGE]), 0)

    def test_a_frame_that_holds_a_registered_page_is_counted(self):
        self.assertEqual(self._count([1 * PAGE, 3 * PAGE]), 1)

    def test_a_frame_that_kept_a_flag_is_counted(self):
        self.assertEqual(self._count([2 * PAGE]), 1)
        self.frame_lp[1] = -1  # released by its page, its exclusive flag left behind
        self.assertEqual(self._count([1 * PAGE]), 1)
        self.assertEqual(self._count([1 * PAGE], exclusive=None), 0)  # no such table without exclusive placement

    def test_the_padding_page_and_rows_outside_the_frames_are_counted(self):
        self.assertEqual(self._count([0, 5 * PAGE, 3 * PAGE]), 2)


class TestReleaseGuard(CustomTestCase):
    """Five frames of four rows, host rows from row 24. The ending request's pages 20, 21 and 22 are
    registered on frames 1 and 2 and on host."""

    HOST_ROW0 = 24

    def setUp(self):
        super().setUp()
        self.frame_lp = torch.tensor([-1, 20, 21, 30, -1])
        self.first = torch.tensor([20 * PAGE, 21 * PAGE, 22 * PAGE])

    def _astray(self, dev_first):
        return hp.pages_not_in_their_frames(
            first=self.first,
            dev_first=torch.tensor(dev_first),
            frame_lp=self.frame_lp,
            page_size=PAGE,
            host_row0=self.HOST_ROW0,
        ).tolist()

    def test_pages_in_their_frames_and_pages_on_host_pass(self):
        self.assertEqual(self._astray([1 * PAGE, 2 * PAGE, 24 + 8]), [False, False, False])

    def test_a_page_mapped_to_another_page_s_frame_is_astray(self):
        self.assertEqual(self._astray([1 * PAGE, 3 * PAGE, 24 + 8]), [False, True, False])

    def test_a_page_mapped_to_a_frame_nothing_is_registered_on_is_astray(self):
        self.assertEqual(self._astray([4 * PAGE, 2 * PAGE, 24 + 8]), [True, False, False])

    def test_the_padding_page_is_astray(self):
        self.assertEqual(self._astray([1 * PAGE, 0, 24 + 8]), [False, True, False])

    def test_a_row_between_the_frames_and_the_host_rows_is_astray(self):
        # what an eviction writes for a page whose host slot is -1: host_row0 - 1
        self.assertEqual(self._astray([1 * PAGE, 2 * PAGE, 23]), [False, False, True])

    def test_release_raises_before_it_indexes_the_frame_tables(self):
        paging = _paging()
        paging.exclusive = False
        paging.host_row0 = self.HOST_ROW0
        paging.num_device_pages = 5
        paging.frame_lp = self.frame_lp
        paging.registered_len = {1: 3 * PAGE}
        req_to_token = torch.zeros((2, 4 * PAGE), dtype=torch.int32)
        req_to_token[1, : 3 * PAGE] = torch.arange(20 * PAGE, 23 * PAGE)
        mapping = torch.zeros(32 * PAGE, dtype=torch.int32)
        mapping[20 * PAGE] = 1 * PAGE
        mapping[21 * PAGE] = 2 * PAGE
        mapping[22 * PAGE] = 23  # between the frames and the host rows
        paging._coordinator = SimpleNamespace(
            page_size=PAGE,
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            token_to_kv_pool_allocator=SimpleNamespace(
                full_to_hisparse_device_index_mapping=mapping
            ),
        )
        with self.assertRaisesRegex(
            RuntimeError, "not in the frames registered for them: 1 .request slot 1 with 3 registered pages"
        ):
            paging.release(1)
        self.assertEqual(self.frame_lp.tolist(), [-1, 20, 21, 30, -1])
