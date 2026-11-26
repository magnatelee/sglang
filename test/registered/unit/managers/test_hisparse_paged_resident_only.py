"""SGLANG_HISPARSE_PAGED_RESIDENT_ONLY: no request is ever buffered, so the speculative swap keeps
no tables and no scratch, only the all-resident decode graph exists, and a decode or verify
forward without a graph takes the all-resident path too."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.graph_variants import (
    HISPARSE_MIXED,
    HISPARSE_RESIDENT,
    HiSparseResidentGraphVariants,
)
from sglang.srt.managers import hisparse_coordinator as hc
from sglang.srt.managers.hisparse_coordinator import (
    HiSparseCoordinator,
    HiSparseSpecSwapManager,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _swap_coordinator(*, top_k: int = 2048, device_buffer_size: int = 2048):
    return MagicMock(
        token_to_kv_pool_allocator=SimpleNamespace(),
        mem_pool_device=SimpleNamespace(layer_num=2),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.empty((3, 16))),
        device="cpu",
        is_dsv4_hisparse=False,
        top_k=top_k,
        device_buffer_size=device_buffer_size,
    )


def _req(req_pool_idx: int):
    return SimpleNamespace(kv=SimpleNamespace(req_pool_idx=req_pool_idx))


def _batch(req_pool_indices):
    return SimpleNamespace(
        reqs=[_req(r) for r in req_pool_indices],
        req_pool_indices=torch.tensor(req_pool_indices),
        out_cache_loc=torch.zeros(len(req_pool_indices) * 6, dtype=torch.int64),
        seq_lens=torch.full((len(req_pool_indices),), 100),
        forward_mode=SimpleNamespace(is_idle=lambda: False),
    )


def _coordinator(*, resident_only: bool, resident=(0, 1)):
    coordinator = object.__new__(HiSparseCoordinator)
    coordinator.enable_resident = True
    coordinator.hs_resident_only = resident_only
    coordinator._resident = {r: object() for r in resident}
    return coordinator


class TestResidentOnlySpecSwap(CustomTestCase):
    def test_no_swap_state_is_built(self):
        spec_swap = HiSparseSpecSwapManager(
            _swap_coordinator(), num_draft_tokens=6, resident_only=True
        )
        self.assertTrue(spec_swap.enabled)
        self.assertFalse(spec_swap.has_swap_state)
        self.assertEqual(spec_swap.scratch_capacity, 0)
        self.assertEqual(spec_swap.states, ())
        self.assertIsNone(spec_swap.top_k_device_locs)
        self.assertEqual(spec_swap.req_to_scratch.shape, (2, 3, 0))

    def test_the_swap_kernel_limits_do_not_apply(self):
        # The buffered path refuses both: more verify positions than its kernel takes, a short top-k.
        for kwargs in ({"num_draft_tokens": 9}, {"num_draft_tokens": 6}):
            coordinator = _swap_coordinator(top_k=512)
            with self.assertRaises(ValueError):
                HiSparseSpecSwapManager(coordinator, **kwargs)
            HiSparseSpecSwapManager(coordinator, resident_only=True, **kwargs)

    def test_default_keeps_the_buffered_path(self):
        spec_swap = HiSparseSpecSwapManager(_swap_coordinator(), num_draft_tokens=6)
        self.assertTrue(spec_swap.has_swap_state)
        self.assertGreater(spec_swap.scratch_capacity, 0)
        self.assertFalse(HiSparseSpecSwapManager.resident_only)

    def test_scratch_calls_take_nothing(self):
        coordinator = _swap_coordinator()
        spec_swap = HiSparseSpecSwapManager(
            coordinator, num_draft_tokens=6, resident_only=True
        )
        owned = torch.arange(4)
        spec_swap.allocate_scratch(0)
        spec_swap.reset(0)
        spec_swap.invalidate_cache(0)
        spec_swap.free_unrotated_scratch(0)
        spec_swap.clear_scratch(0)
        self.assertIs(spec_swap.extend_owned_locs(0, owned), owned)
        self.assertEqual(spec_swap.spare_scratch_slots(), 0)
        self.assertFalse(spec_swap.free_spare_scratch())
        coordinator.reclaim_device.assert_not_called()

    def test_verify_of_resident_requests_selects_the_resident_graph(self):
        coordinator = _swap_coordinator()
        coordinator._hs_spec_res_prepare.return_value = [0, 1]
        coordinator.page_size = 4  # fewer usable slots than verify positions: no extra page is used
        spec_swap = HiSparseSpecSwapManager(
            coordinator, num_draft_tokens=6, resident_only=True
        )
        spec_swap.prepare_verify(_batch([0, 1]))
        self.assertEqual(coordinator._hs_verify_sel, 2)

    def test_verify_with_a_buffered_request_raises(self):
        coordinator = _swap_coordinator()
        coordinator._hs_spec_res_prepare.return_value = [0]
        spec_swap = HiSparseSpecSwapManager(
            coordinator, num_draft_tokens=6, resident_only=True
        )
        with self.assertRaisesRegex(RuntimeError, "not resident"):
            spec_swap.prepare_verify(_batch([0, 1]))

    def test_commit_of_resident_requests_copies_nothing(self):
        coordinator = _swap_coordinator()
        coordinator.hs_spec_res = True
        coordinator._resident = {0: object(), 1: object()}
        spec_swap = HiSparseSpecSwapManager(
            coordinator, num_draft_tokens=6, resident_only=True
        )
        spec_swap.commit_accept_tokens(_batch([0, 1]), torch.zeros((2, 6)))
        coordinator._gpu_rows.assert_not_called()
        coordinator._resident = {0: object()}
        with self.assertRaisesRegex(RuntimeError, "not resident"):
            spec_swap.commit_accept_tokens(_batch([0, 1]), torch.zeros((2, 6)))


class TestResidentOnlyCoordinator(CustomTestCase):
    def test_class_default_is_off(self):
        self.assertFalse(HiSparseCoordinator.hs_resident_only)

    def test_resident_path_follows_the_graph_label_when_off(self):
        coordinator = _coordinator(resident_only=False)
        with patch.object(hc, "get_capture_attention_variant", return_value=None):
            self.assertFalse(coordinator.resident_variant_active(decode_shaped=True))
        with patch.object(
            hc, "get_capture_attention_variant", return_value=HISPARSE_MIXED
        ):
            self.assertFalse(coordinator.resident_variant_active(decode_shaped=True))
        with patch.object(
            hc, "get_capture_attention_variant", return_value=HISPARSE_RESIDENT
        ):
            self.assertTrue(coordinator.resident_variant_active(decode_shaped=True))
            self.assertTrue(coordinator.resident_variant_active(decode_shaped=False))

    def test_decode_without_a_graph_takes_the_resident_path(self):
        coordinator = _coordinator(resident_only=True)
        with patch.object(hc, "get_capture_attention_variant", return_value=None):
            self.assertTrue(coordinator.resident_variant_active(decode_shaped=True))
            # A prefill keeps its own translation.
            self.assertFalse(coordinator.resident_variant_active(decode_shaped=False))
            self.assertIsNone(
                coordinator.resident_page_table(MagicMock(), decode_shaped=False)
            )

    def test_top_k_rows_pass_through_without_a_swap_in(self):
        coordinator = _coordinator(resident_only=True)
        coordinator._pool_layer_index = None
        coordinator._run_swap_in_kernel = MagicMock()
        rows = torch.arange(8, dtype=torch.int32).view(2, 4)
        with patch.object(hc, "get_capture_attention_variant", return_value=None):
            out = coordinator._hs_swap_in_selected_pages(
                torch.tensor([0, 1]), torch.tensor([100, 100]), rows, 0
            )
        self.assertIs(out, rows)
        coordinator._run_swap_in_kernel.assert_not_called()

    def test_a_buffered_request_in_a_decode_batch_raises(self):
        seq_lens = torch.tensor([10, 11])
        coordinator = _coordinator(resident_only=True, resident=(0,))
        with self.assertRaisesRegex(RuntimeError, "not resident"):
            coordinator._hs_build_dev_page_table(None, seq_lens, torch.tensor([0, 1]))
        # Off: such a batch is just not all-resident. On: an idle batch is no error.
        mixed = _coordinator(resident_only=False, resident=(0,))
        mixed._hs_build_dev_page_table(None, seq_lens, torch.tensor([0, 1]))
        self.assertIsNone(mixed._hs_pt_batch)
        coordinator._hs_build_dev_page_table(None, seq_lens[:0], torch.tensor([]))
        self.assertIsNone(coordinator._hs_pt_batch)
        coordinator._hs_build_dev_page_table(None, seq_lens[:1], torch.tensor([0]))
        self.assertEqual(coordinator._hs_pt_batch, (0,))


class TestResidentOnlyGraphVariants(CustomTestCase):
    def test_only_the_resident_graph_is_captured_and_selected(self):
        coordinator = MagicMock(hs_resident_only=True)
        coordinator.hs_batch_all_resident.return_value = False  # an idle batch
        variants = HiSparseResidentGraphVariants(coordinator)
        self.assertEqual(variants.capture_labels, (HISPARSE_RESIDENT,))
        self.assertEqual(variants.select(MagicMock()), HISPARSE_RESIDENT)
        coordinator.hs_batch_all_resident.assert_called_once()

    def test_both_graphs_when_off(self):
        coordinator = MagicMock(hs_resident_only=False)
        variants = HiSparseResidentGraphVariants(coordinator)
        self.assertEqual(variants.capture_labels, (HISPARSE_MIXED, HISPARSE_RESIDENT))
        coordinator.hs_batch_all_resident.return_value = False
        self.assertEqual(variants.select(MagicMock()), HISPARSE_MIXED)
        coordinator.hs_batch_all_resident.return_value = True
        self.assertEqual(variants.select(MagicMock()), HISPARSE_RESIDENT)


if __name__ == "__main__":
    unittest.main()
