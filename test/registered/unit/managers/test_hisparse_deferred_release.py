import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers import hisparse_coordinator as hc
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Event:
    """A stream event the test completes by hand."""

    def __init__(self):
        self.stream = None
        self.done = False
        self.synchronized = False

    def record(self, stream=None):
        self.stream = stream

    def query(self):
        return self.done

    def synchronize(self):
        self.synchronized = True
        self.done = True


class _Stream:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def wait_stream(self, stream):
        self.log.append((self.name, "waits for", stream.name))


class _DeviceModule:
    """Events the test completes, and the stream that is current."""

    def __init__(self, log):
        self.events = []
        self.current = _Stream("scheduler", log)

    def Event(self):
        self.events.append(_Event())
        return self.events[-1]

    def current_stream(self):
        return self.current

    @contextlib.contextmanager
    def stream(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous


def _coordinator(calls, device):
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator.enable_resident = True
    coordinator.decode_producer_stream = "forward"
    coordinator.decode_backup_stream = "backup"
    coordinator._release_queue = []
    coordinator._release_pending = set()
    coordinator._release_stream = _Stream("release", calls)
    coordinator.release_request = lambda req, fenced=False: calls.append(
        ("hisparse", req.rid, fenced, device.current.name)
    )
    return coordinator


def _req(rid, slot):
    return SimpleNamespace(
        rid=rid, kv=SimpleNamespace(req_pool_idx=slot), hisparse_staging=False
    )


class TestHiSparseDeferredRelease(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.calls = []
        self.device = _DeviceModule(self.calls)
        self.events = self.device.events
        patcher = patch.object(hc, "device_module", self.device)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _coordinator(self):
        return _coordinator(self.calls, self.device)

    def test_a_request_still_staging_is_released_at_once(self):
        # Its prompt is being copied to host and no forward has read its rows:
        # the caller releases it now, as before.
        coordinator = self._coordinator()
        req = _req("s", 5)
        req.hisparse_staging = True
        self.assertFalse(self._defer(coordinator, req, self.calls))
        self.assertEqual(coordinator._release_queue, [])

    def _defer(self, coordinator, req, calls):
        return coordinator.defer_release(
            req, release=lambda: calls.append(("row", req.rid))
        )

    def test_release_waits_for_the_forward_and_backup_streams(self):
        calls = self.calls
        coordinator = self._coordinator()
        self.assertTrue(self._defer(coordinator, _req("a", 3), calls))
        self.assertEqual([e.stream for e in self.events[:2]], ["forward", "backup"])
        self.events[2].done = True

        coordinator.run_deferred_releases()
        self.assertEqual(calls, [])
        self.events[0].done = True
        coordinator.run_deferred_releases()
        self.assertEqual(calls, [])  # the backup of its last tokens is still copying

        self.events[1].done = True
        coordinator.run_deferred_releases()
        # On the release stream, which the scheduler's stream then follows: a
        # host read there does not wait for the forward the scheduler waits for.
        self.assertEqual(
            calls,
            [
                ("hisparse", "a", True, "release"),
                ("row", "a"),
                ("scheduler", "waits for", "release"),
            ],
        )
        self.assertEqual(coordinator._release_queue, [])
        self.assertEqual(coordinator._release_pending, set())

    def test_release_waits_for_what_the_ending_stream_has_queued(self):
        """A request aborted in the pass it was pre-allocated in: its row and map entries are still queued
        on the scheduler's stream, and a release that read them now would free the row's previous occupant."""
        calls = self.calls
        coordinator = self._coordinator()
        self.assertTrue(self._defer(coordinator, _req("a", 3), calls))
        self.assertEqual(self.events[2].stream.name, "scheduler")
        self.events[0].done = self.events[1].done = True

        coordinator.run_deferred_releases()
        self.assertEqual(calls, [])

        self.events[2].done = True
        coordinator.run_deferred_releases()
        self.assertEqual(calls[:2], [("hisparse", "a", True, "release"), ("row", "a")])

    def test_release_waits_for_an_eviction_that_has_not_run(self):
        """An eviction's picks and map writes run when its stream gets to them; a release beside it frees the
        frames of the same pages a second time."""
        calls = self.calls
        coordinator = self._coordinator()
        evicting = _Event()
        coordinator.paging = SimpleNamespace(
            seal_fence=lambda: None,
            unlist=lambda rid: None,
            evict_fence=lambda: None if evicting.done else evicting,
        )
        self.assertTrue(self._defer(coordinator, _req("a", 3), calls))
        for event in self.events:
            event.done = True

        coordinator.run_deferred_releases()
        self.assertEqual(calls, [])
        self.assertFalse(evicting.synchronized)

        evicting.done = True
        coordinator.run_deferred_releases()
        self.assertEqual(calls[:2], [("hisparse", "a", True, "release"), ("row", "a")])

    def test_blocking_release_waits_for_the_eviction(self):
        calls = self.calls
        coordinator = self._coordinator()
        evicting = _Event()
        coordinator.paging = SimpleNamespace(
            seal_fence=lambda: None,
            unlist=lambda rid: None,
            evict_fence=lambda: None if evicting.done else evicting,
        )
        self.assertTrue(self._defer(coordinator, _req("a", 3), calls))
        for event in self.events:
            event.done = True

        coordinator.run_deferred_releases(block=True)
        self.assertTrue(evicting.synchronized)
        self.assertEqual(calls[:2], [("hisparse", "a", True, "release"), ("row", "a")])

    def test_requests_are_released_in_the_order_they_ended(self):
        calls = self.calls
        coordinator = self._coordinator()
        self._defer(coordinator, _req("a", 3), calls)
        self._defer(coordinator, _req("b", 5), calls)
        for event in self.events[3:]:  # b's users have run, a's have not
            event.done = True
        coordinator.run_deferred_releases()
        self.assertEqual(calls, [])

        for event in self.events:
            event.done = True
        coordinator.run_deferred_releases()
        self.assertEqual(
            [call[1] for call in calls if call[0] == "row"], ["a", "b"]
        )

    def test_a_request_that_ends_twice_is_released_once(self):
        calls = self.calls
        coordinator = self._coordinator()
        req = _req("a", 3)
        self.assertTrue(self._defer(coordinator, req, calls))
        self.assertTrue(self._defer(coordinator, req, calls))
        self.assertEqual(len(coordinator._release_queue), 1)

    def test_blocking_waits_for_the_users_instead_of_skipping_them(self):
        calls = self.calls
        coordinator = self._coordinator()
        self._defer(coordinator, _req("a", 3), calls)
        coordinator.run_deferred_releases(block=True)
        self.assertTrue(all(event.synchronized for event in self.events))
        self.assertEqual(calls[:2], [("hisparse", "a", True, "release"), ("row", "a")])

    def test_nothing_is_deferred_without_paged_residency_or_an_overlap_stream(self):
        calls = self.calls
        coordinator = self._coordinator()
        coordinator.enable_resident = False
        self.assertFalse(self._defer(coordinator, _req("a", 3), calls))
        coordinator.enable_resident = True
        coordinator.decode_producer_stream = None
        self.assertFalse(self._defer(coordinator, _req("a", 3), calls))
        self.assertEqual(coordinator._release_queue, [])

    def test_a_coordinator_built_without_init_has_nothing_to_release(self):
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.run_deferred_releases(block=True)


if __name__ == "__main__":
    unittest.main()
