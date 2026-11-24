"""Sizes of the HiSparse speculative swap at the kernel's limits (eight draft tokens)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseSpecSwapManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _coordinator(*, top_k: int, device_buffer_size: int):
    return MagicMock(
        token_to_kv_pool_allocator=SimpleNamespace(),
        mem_pool_device=SimpleNamespace(layer_num=2),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.empty((3, 16))),
        device="cpu",
        is_dsv4_hisparse=False,
        top_k=top_k,
        device_buffer_size=device_buffer_size,
    )


class TestHiSparseSpecLimits(CustomTestCase):
    def test_eight_draft_tokens_fill_the_workspace_exactly(self):
        """Seven draft steps verify eight positions: 8 x 2048 occurrences, the kernel's
        limit. The hot buffer and a request's scratch rows together hold a round in
        which every occurrence is a distinct position."""
        for buffer_size, scratch in ((12288, 4096), (8192, 8192), (16384, 1024)):
            with self.subTest(buffer_size=buffer_size):
                spec_swap = HiSparseSpecSwapManager(
                    _coordinator(top_k=2048, device_buffer_size=buffer_size),
                    num_draft_tokens=8,
                )
                self.assertTrue(spec_swap.enabled)
                self.assertEqual(spec_swap.scratch_capacity, scratch)
                self.assertGreaterEqual(buffer_size + scratch, 8 * 2048)
                self.assertEqual(spec_swap.miss_plan_capacity, 8 * 2048)
                self.assertEqual(spec_swap.top_k_device_locs.shape, (3, 8, 2048))
                self.assertEqual(spec_swap.req_to_scratch.shape, (2, 3, scratch))

    def test_more_than_the_kernel_takes_is_refused(self):
        with self.assertRaisesRegex(ValueError, "draft tokens"):
            HiSparseSpecSwapManager(
                _coordinator(top_k=2048, device_buffer_size=12288), num_draft_tokens=9
            )
        with self.assertRaisesRegex(ValueError, "occurrences"):
            HiSparseSpecSwapManager(
                _coordinator(top_k=4096, device_buffer_size=12288), num_draft_tokens=8
            )


if __name__ == "__main__":
    unittest.main()
