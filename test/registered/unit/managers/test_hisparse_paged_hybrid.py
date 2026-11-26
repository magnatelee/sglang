"""Paged HiSparse on a hybrid linear-attention model (Kimi-K3.1): the DSA layers live in the inner
pool of a HybridLinearKVPool, so the allocator takes the mapped ranges from that pool and the
paging tables, which are indexed by pool layer, get the pool's layer for a model layer id."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 64


def _hybrid_pool(ranges):
    inner = object.__new__(HiSparseDSATokenToKVPool)
    inner._hs_el = ranges
    inner.kv_buffer = [torch.arange(8.0).view(8, 1, 1).clone() for _ in range(2)]
    inner.full_to_hisparse_device_index_mapping = None
    hybrid = object.__new__(HybridLinearKVPool)
    hybrid.full_kv_pool = inner
    hybrid.full_attention_layer_id_mapping = {3: 0, 7: 1}
    return hybrid, inner


def _coordinator(pool_layer_index):
    coordinator = object.__new__(HiSparseCoordinator)
    coordinator.enable_resident = True
    coordinator._pool_layer_index = pool_layer_index
    coordinator.paging = MagicMock()
    coordinator.paging.redirect.side_effect = lambda locs, layer: locs
    coordinator.paging.read_through_row_cache.side_effect = lambda locs, layer: locs
    coordinator._hs_swap_in_selected_pages = MagicMock(
        return_value=torch.zeros((1, 4), dtype=torch.int32)
    )
    return coordinator


class TestPagedHybridPool(CustomTestCase):
    def test_allocator_takes_the_ranges_of_the_inner_pool(self):
        ranges = {"k_max": 4 * PAGE}
        hybrid, inner = _hybrid_pool(ranges)

        def limits(allocator):
            allocator._hs_el.update(dev_max_page=1 << 30, log_max_page=1 << 30)

        with patch.object(
            HiSparseTokenToKVPoolAllocator, "_hs_el_init_limits", limits
        ):
            allocator = HiSparseTokenToKVPoolAllocator(
                size=2 * PAGE,
                page_size=PAGE,
                dtype=torch.float8_e4m3fn,
                device="cpu",
                kvcache=hybrid,
                need_sort=False,
                host_to_device_ratio=2,
            )
        self.assertIs(allocator._hs_el, ranges)
        self.assertIs(allocator.get_kvcache(), hybrid)
        # The device allocator spans the band the ranges name.
        self.assertEqual(allocator.hisparse_attn_allocator.size, 4 * PAGE)
        self.assertIsNotNone(inner.full_to_hisparse_device_index_mapping)

    def test_a_pool_without_ranges_keeps_main_behaviour(self):
        hybrid, _ = _hybrid_pool(None)
        allocator = HiSparseTokenToKVPoolAllocator(
            size=2 * PAGE,
            page_size=PAGE,
            dtype=torch.float8_e4m3fn,
            device="cpu",
            kvcache=hybrid,
            need_sort=False,
            host_to_device_ratio=2,
        )
        self.assertIsNone(allocator._hs_el)
        self.assertEqual(allocator.hisparse_attn_allocator.size, 2 * PAGE)

    def test_row_copies_go_to_the_inner_pool(self):
        hybrid, inner = _hybrid_pool(None)
        allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
        allocator._kvcache = hybrid
        allocator.hs_copy_rows(torch.tensor([1, 2]), torch.tensor([5, 6]))
        for buf in inner.kv_buffer:
            self.assertEqual(buf.view(-1).tolist(), [0, 1, 2, 3, 4, 1, 2, 7])

    def test_pool_layer_of(self):
        self.assertIsNone(HiSparseCoordinator._pool_layer_index)
        self.assertEqual(_coordinator(None).pool_layer_of(5), 5)
        coordinator = _coordinator({3: 0, 7: 1})
        self.assertEqual(coordinator.pool_layer_of(3), 0)
        self.assertEqual(coordinator.pool_layer_of(7), 1)
        with self.assertRaises(KeyError):
            coordinator.pool_layer_of(4)  # a linear-attention layer holds no KV

    def test_paging_gets_the_pool_layer(self):
        coordinator = _coordinator({3: 0, 7: 1})
        req_pool_indices = torch.zeros(1, dtype=torch.int64)
        seq_lens = torch.full((1,), 100, dtype=torch.int32)
        top_k = torch.zeros((1, 4), dtype=torch.int32)
        coordinator.swap_in_selected_pages(req_pool_indices, seq_lens, top_k, 7)
        # The swap-in keeps the model's id (it maps it itself); the paging tables get the pool's.
        self.assertEqual(coordinator._hs_swap_in_selected_pages.call_args.args[3], 7)
        self.assertEqual(coordinator.paging.redirect.call_args.args[1], 1)
        self.assertEqual(coordinator.paging.read_through_row_cache.call_args.args[1], 1)

    def test_without_paged_residency_paging_is_not_called(self):
        coordinator = _coordinator({3: 0, 7: 1})
        coordinator.enable_resident = False
        coordinator.swap_in_selected_pages(
            torch.zeros(1, dtype=torch.int64),
            torch.full((1,), 100, dtype=torch.int32),
            torch.zeros((1, 4), dtype=torch.int32),
            3,
        )
        coordinator.paging.redirect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
