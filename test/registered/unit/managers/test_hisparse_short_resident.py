"""Paged HiSparse with short prompts resident (SGLANG_HISPARSE_PAGED_SHORT_RESIDENT): a prompt no
longer than the hot buffer keeps its prefill pages like every longer one, so nothing charges, holds
or builds a hot buffer for it."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.hisparse_paging import HiSparsePagingManager
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BUFFER = 2048
PAGE = 64


def _coordinator(short_resident: bool) -> HiSparseCoordinator:
    coordinator = object.__new__(HiSparseCoordinator)
    coordinator.enable_resident = True
    coordinator.hs_exclusive = True
    coordinator.hs_short_resident = short_resident
    coordinator.device_buffer_size = BUFFER
    coordinator.is_dsv4_hisparse = False
    coordinator.page_size = PAGE
    return coordinator


def _req(length: int, req_pool_idx: int = 3):
    return SimpleNamespace(
        kv=SimpleNamespace(kv_allocated_len=length, req_pool_idx=req_pool_idx),
        full_untruncated_fill_ids=[0] * length,
    )


class TestShortPromptsResident(CustomTestCase):
    def test_switch_is_off_by_default(self):
        self.assertFalse(HiSparseCoordinator.hs_short_resident)
        coordinator = object.__new__(HiSparseCoordinator)
        coordinator.device_buffer_size = BUFFER
        self.assertEqual(coordinator._buffered_prompt_limit(), BUFFER)

    def test_a_prompt_of_any_length_stays_resident_with_the_switch(self):
        off, on = _coordinator(False), _coordinator(True)
        for length in (1, 25, 63, 64, 65, 2047, 2048):
            self.assertFalse(off._can_stay_resident(_req(length)), length)
            self.assertTrue(on._can_stay_resident(_req(length)), length)
        for length in (2049, 200_000):
            self.assertTrue(off._can_stay_resident(_req(length)), length)
            self.assertTrue(on._can_stay_resident(_req(length)), length)

    def test_without_paged_residency_nothing_is_resident(self):
        coordinator = _coordinator(True)
        coordinator.enable_resident = False
        self.assertFalse(coordinator._can_stay_resident(_req(100)))
        self.assertFalse(coordinator._can_stay_resident(_req(100_000)))

    def test_a_short_prompt_reserves_no_buffered_host_rows(self):
        """With a host pool below the logical pool a prompt that stays buffered
        takes its host rows with its request slot; a resident one has none."""
        for short_resident, expected_calls in ((False, 1), (True, 0)):
            coordinator = _coordinator(short_resident)
            coordinator.hs_host_scarce = True
            coordinator.req_to_host_pool = object()
            coordinator.req_to_host_pool_allocated_len = torch.zeros(
                8, dtype=torch.int64
            )
            coordinator.paging = SimpleNamespace(ensure_host=MagicMock())
            coordinator.mem_pool_host = SimpleNamespace(
                alloc_paged_token_slots=MagicMock()
            )

            coordinator.reserve_buffered_host_rows([_req(300), _req(5000, 4)])

            self.assertEqual(
                coordinator.paging.ensure_host.call_count, expected_calls
            )
            self.assertEqual(
                coordinator.mem_pool_host.alloc_paged_token_slots.call_count,
                expected_calls,
            )

    def test_no_buffer_is_grown_for_a_short_resident_request(self):
        """Plain decode's general path: a request no longer than the buffer with
        no buffer yet would get one grown, unless short prompts are resident."""
        seq_lens = torch.tensor([300, 5000], dtype=torch.int64)
        slots = torch.tensor([3, 4], dtype=torch.int64)
        for short_resident, expected_allocs in ((False, 1), (True, 0)):
            coordinator = _coordinator(short_resident)
            coordinator.padded_buffer_size = BUFFER + PAGE
            coordinator.req_device_buffer_size = torch.zeros(8, dtype=torch.int64)
            coordinator.req_to_device_buffer = torch.zeros(
                (8, BUFFER + PAGE), dtype=torch.int64
            )
            coordinator.req_device_buffer_token_locs = torch.zeros(
                (1, 8, BUFFER + PAGE), dtype=torch.int32
            )
            coordinator.mem_pool_device = SimpleNamespace(page_size=PAGE)
            coordinator.hs_el = None
            alloc = MagicMock(side_effect=lambda need: torch.arange(64, 64 + need))
            coordinator.token_to_kv_pool_allocator = SimpleNamespace(
                hisparse_attn_allocator=SimpleNamespace(alloc=alloc)
            )

            coordinator._grow_device_buffers(seq_lens, slots, seq_lens, slots)

            self.assertEqual(alloc.call_count, expected_allocs)
            self.assertEqual(
                int(coordinator.req_device_buffer_size[3]),
                0 if short_resident else 320,
            )


class TestShortResidentRequestEnds(CustomTestCase):
    def test_a_request_without_a_registered_page_leaves_no_staged_length(self):
        """A resident request shorter than a page ends before any whole page of it
        was registered; its staged length must not outlive it in the slot."""
        paging = object.__new__(HiSparsePagingManager)
        paging._coordinator = SimpleNamespace()
        paging.exclusive = True
        paging._seal_unlisted = {}
        paging._seal_pending = None
        paging.finish_seal = MagicMock()
        paging.registered_len = {}
        paging.host_len = {3: 0, 4: 128}

        paging.release(3)

        self.assertEqual(paging.host_len, {4: 128})


class TestShortPromptsResidentAllocator(CustomTestCase):
    def _allocator(self, device_buffer_size: int) -> HiSparseTokenToKVPoolAllocator:
        allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
        allocator.page_size = PAGE
        allocator.device = "cpu"
        allocator._hs_el = None
        allocator._reclaimer = None
        allocator.hisparse_attn_allocator = SimpleNamespace(
            available_size=lambda: 1 << 20,
            alloc=MagicMock(side_effect=lambda need: torch.arange(64, 64 + need)),
        )
        allocator.configure_spec_scratch(1024, device_buffer_size=device_buffer_size)
        return allocator

    def test_a_short_prompt_is_charged_its_scratch_and_no_buffer(self):
        """The coordinator configures the allocator with no buffer size when short
        prompts are resident: admission and pre-allocation then price a short
        prompt like a long one."""
        buffered = self._allocator(BUFFER + PAGE)
        resident = self._allocator(0)

        def reserve(allocator, prefill_tokens, has_req_pool_slot=False):
            return allocator.request_slot_reserve(
                has_req_pool_slot=has_req_pool_slot, prefill_tokens=prefill_tokens
            )

        self.assertEqual(reserve(buffered, 300), BUFFER + PAGE - 320 + 1024)
        self.assertEqual(reserve(resident, 300), 1024)
        self.assertEqual(reserve(resident, 300, has_req_pool_slot=True), 0)
        self.assertEqual(reserve(resident, 1), 1024)
        self.assertEqual(reserve(resident, 200_000), reserve(buffered, 200_000))

    def test_a_short_prompt_holds_no_buffer_slots_from_pre_allocation(self):
        buffered = self._allocator(BUFFER + PAGE)
        resident = self._allocator(0)

        self.assertEqual(buffered.alloc_buffer_extra(300).numel(), BUFFER + PAGE - 320)
        self.assertEqual(resident.alloc_buffer_extra(300).numel(), 0)
        self.assertEqual(resident.alloc_buffer_extra(1).numel(), 0)
        resident.hisparse_attn_allocator.alloc.assert_not_called()

        coordinator = object.__new__(HiSparseCoordinator)
        coordinator.token_to_kv_pool_allocator = resident
        self.assertTrue(coordinator.hold_buffer_slots(_req(300), 300))
        self.assertIsNone(coordinator._recv_buffer_hold)


if __name__ == "__main__":
    unittest.main()
