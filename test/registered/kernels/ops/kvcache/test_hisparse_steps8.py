"""HiSparse slot tables with eight verify positions per request (MTP with 7 draft steps).

The kernels take the number of verify positions as a parameter; the upstream tests
run them at one to four positions (one swap test at eight). These run the resident
kernels and the speculative swap at eight positions and top-k 2048, the shape a
GLM-5.2 verify batch has with 7 draft steps.
"""

import importlib.util
import pathlib
import unittest

import pytest
import torch

from sglang.srt.utils import is_npu, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or is_npu() or is_xpu(),
    reason="HiSparse kernel tests require a CUDA GPU.",
)

NUM_STEPS = 8
NUM_TOP_K = 2048
NUM_REQUEST_SLOTS = 6
MAX_CONTEXT_LEN = 20000


def _swap_helpers():
    """The state builder and runner of the speculative swap tests next to this file."""
    path = pathlib.Path(__file__).with_name("test_hisparse_spec.py")
    spec = importlib.util.spec_from_file_location("_hisparse_spec_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request_tables():
    """Request slots 1 and 4 are resident. Rows go up to the int32 limit, where the
    host rows of a mapped address range sit."""
    generator = torch.Generator().manual_seed(0)
    num_logical = NUM_REQUEST_SLOTS * MAX_CONTEXT_LEN
    req_to_token = torch.randperm(num_logical, generator=generator, dtype=torch.int32)
    req_to_token = req_to_token.view(NUM_REQUEST_SLOTS, MAX_CONTEXT_LEN).cuda()
    mapping = torch.randperm(num_logical, generator=generator) + (2**31 - 1 - num_logical)
    resident = torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.int8, device="cuda")
    return req_to_token, mapping.cuda(), resident


def _verify_seq_lens(num_reqs: int) -> torch.Tensor:
    """Step s of a request sees its length before the round plus s + 1."""
    base = torch.randint(NUM_STEPS + 1, MAX_CONTEXT_LEN - NUM_STEPS, (num_reqs, 1))
    return (base + torch.arange(1, NUM_STEPS + 1)).to(torch.int32).reshape(-1).cuda()


class TestHiSparseResidentRowsEightSteps(CustomTestCase):
    def _expected(self, top_k, seq_lens, req_pool_indices, tables, fill):
        req_to_token, mapping, resident = tables
        num_reqs = req_pool_indices.numel()
        positions = top_k.reshape(num_reqs, NUM_STEPS, NUM_TOP_K).long()
        lengths = seq_lens.view(num_reqs, NUM_STEPS, 1).long()
        selected = (positions >= 0) & (positions < lengths)
        logical = req_to_token[
            req_pool_indices.view(-1, 1, 1), positions.clamp(min=0)
        ].long()
        rewritten = (
            resident[req_pool_indices].bool()
            & (torch.arange(num_reqs, device="cuda") < num_reqs - 1)
        ).view(-1, 1, 1)
        return torch.where(
            rewritten, torch.where(selected, mapping[logical], -1), fill
        ).to(torch.int32)

    def test_translation_at_eight_steps_and_top_k_2048(self):
        from sglang.kernels.ops.kvcache.hisparse_resident import translate_resident_rows

        tables = _request_tables()
        req_pool_indices = torch.tensor([4, 0, 1, 2, 1, 4], device="cuda")
        num_reqs = req_pool_indices.numel()
        torch.manual_seed(1)
        top_k = torch.randint(
            -2, MAX_CONTEXT_LEN, (num_reqs, NUM_STEPS, NUM_TOP_K), dtype=torch.int32
        ).cuda()
        seq_lens = _verify_seq_lens(num_reqs)
        output = torch.full_like(top_k, -7)
        # The last batch row is CUDA-graph padding although its slot is resident.
        num_real_reqs = torch.tensor([num_reqs - 1], dtype=torch.int32, device="cuda")

        translate_resident_rows(
            top_k=top_k,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            resident=tables[2],
            req_to_token=tables[0],
            mapping=tables[1],
            num_real_reqs=num_real_reqs,
            output=output,
        )

        expected = self._expected(top_k, seq_lens, req_pool_indices, tables, -7)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        # Rows of resident requests were really selected and sit near the int32 limit.
        self.assertGreater(int((expected > 2**30).sum()), NUM_TOP_K)

    def test_translation_through_padded_row_strides(self):
        """The indexer hands out slot tables whose rows are wider than top-k (a view
        of a padded buffer): each query's row starts at the table's own stride."""
        from sglang.kernels.ops.kvcache.hisparse_resident import translate_resident_rows

        tables = _request_tables()
        req_pool_indices = torch.tensor([1, 3, 4], device="cuda")
        num_reqs = req_pool_indices.numel()
        torch.manual_seed(2)
        wide = torch.randint(
            -2, MAX_CONTEXT_LEN, (num_reqs, NUM_STEPS, NUM_TOP_K + 64), dtype=torch.int32
        ).cuda()
        wide_out = torch.full_like(wide, -7)
        top_k, output = wide[:, :, :NUM_TOP_K], wide_out[:, :, :NUM_TOP_K]
        seq_lens = _verify_seq_lens(num_reqs)
        num_real_reqs = torch.tensor([num_reqs], dtype=torch.int32, device="cuda")

        translate_resident_rows(
            top_k=top_k,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            resident=tables[2],
            req_to_token=tables[0],
            mapping=tables[1],
            num_real_reqs=num_real_reqs,
            output=output,
        )

        # (the helper treats the last row as padding: compare the first two rows)
        expected = self._expected(top_k, seq_lens, req_pool_indices, tables, -7)
        torch.testing.assert_close(output[:2], expected[:2], rtol=0, atol=0)
        positions = top_k[2].long()
        selected = (positions >= 0) & (positions < seq_lens.view(num_reqs, NUM_STEPS, 1)[2])
        logical = tables[0][4, positions.clamp(min=0)].long()
        torch.testing.assert_close(
            output[2], torch.where(selected, tables[1][logical], -1).to(torch.int32)
        )
        # Nothing was written past a row's top-k entries.
        self.assertTrue(torch.all(wide_out[:, :, NUM_TOP_K:] == -7).item())

    def test_masked_lengths_at_eight_steps(self):
        from sglang.kernels.ops.kvcache.hisparse_resident import mask_resident_seq_lens

        _, _, resident = _request_tables()
        for num_reqs in (1, 7, 48):
            req_pool_indices = torch.randint(0, NUM_REQUEST_SLOTS, (num_reqs,)).cuda()
            for dtype in (torch.int32, torch.int64):
                with self.subTest(num_reqs=num_reqs, dtype=dtype):
                    seq_lens = _verify_seq_lens(num_reqs).to(dtype)
                    original = seq_lens.clone()
                    masked = mask_resident_seq_lens(
                        seq_lens=seq_lens,
                        req_pool_indices=req_pool_indices,
                        resident=resident,
                    )
                    is_resident = resident[req_pool_indices].bool()
                    expected = torch.where(
                        is_resident.repeat_interleave(NUM_STEPS), 0, original
                    )
                    torch.testing.assert_close(masked, expected, rtol=0, atol=0)


class TestHiSparseSpecSwapEightSteps(CustomTestCase):
    """The speculative swap at the size a 12,288-token hot buffer has with eight
    verify positions: 16,384 occurrences, 4,096 scratch rows per request."""

    HOT, PAGE, SCRATCH = 12288, 64, 4096

    def _state(self, helpers, num_reqs, seq_len):
        return helpers._make_state(
            num_reqs=num_reqs,
            hot_buffer_size=self.HOT,
            page_size=self.PAGE,
            scratch_size=self.SCRATCH,
            seq_len=seq_len,
            item_words=1,
            metadata_occurrences=NUM_STEPS * NUM_TOP_K,
        )

    def test_every_occurrence_a_distinct_miss(self):
        """The worst round: eight steps select 16,384 distinct positions, none of them in
        the hot buffer. Hot buffer plus scratch hold exactly that many rows, so every
        hot slot is evicted. With victims probed in 32-bit arithmetic a 12,288-slot
        buffer (not a power of two) left about 30 positions without a row."""
        helpers = _swap_helpers()
        state = self._state(helpers, 1, 32768)
        first = torch.arange(NUM_STEPS, dtype=torch.int32, device="cuda") * NUM_TOP_K
        top_k_tokens = (
            self.HOT
            + first.view(1, -1, 1)
            + torch.arange(NUM_TOP_K, dtype=torch.int32, device="cuda").view(1, 1, -1)
        ).contiguous()
        seq_lens = torch.full((NUM_STEPS,), 32768, dtype=torch.int32, device="cuda")

        out = helpers._run_swap(top_k_tokens=top_k_tokens, seq_lens=seq_lens, state=state)
        torch.cuda.synchronize()

        helpers._assert_output_matches_tokens(state, out, top_k_tokens)
        self.assertEqual(
            int(state.swap_state.scratch_state[0, 0].item()), NUM_STEPS * NUM_TOP_K
        )
        self.assertEqual(int(out.unique().numel()), NUM_STEPS * NUM_TOP_K)

    def test_overlapping_steps_with_recent_positions(self):
        """Consecutive verify positions select nearly the same set: a shared core in
        the hot buffer, a shared set of misses and a few step-specific ones."""
        helpers = _swap_helpers()
        state = self._state(helpers, 2, 65536)
        generator = torch.Generator().manual_seed(3)
        rows = []
        for req in range(2):
            core = torch.randperm(self.HOT, generator=generator)[:1500]
            shared_miss = self.HOT + torch.randperm(40000, generator=generator)[:400]
            steps = []
            for step in range(NUM_STEPS):
                own = self.HOT + 40000 + req * 2000 + step * 148 + torch.arange(148)
                steps.append(torch.cat((core, shared_miss, own)))
            rows.append(torch.stack(steps))
        top_k_tokens = torch.stack(rows).to(torch.int32).cuda().contiguous()
        seq_lens = torch.full((2 * NUM_STEPS,), 65536, dtype=torch.int32, device="cuda")

        out = helpers._run_swap(top_k_tokens=top_k_tokens, seq_lens=seq_lens, state=state)
        torch.cuda.synchronize()

        helpers._assert_output_matches_tokens(state, out, top_k_tokens)
        # A position selected in several steps is copied once and read from one row.
        for req in range(2):
            self.assertEqual(
                int(out[req].unique().numel()), int(top_k_tokens[req].unique().numel())
            )
            self.assertEqual(
                int(state.swap_state.scratch_state[0, req].item()), 400 + NUM_STEPS * 148
            )

    def test_rounds_that_need_nearly_every_hot_slot(self):
        """Rounds whose selections cover about as many positions as the hot buffer
        holds, with more misses than scratch rows: most victims come from the
        exhaustive probe. Without a barrier between the eviction iterations a warp's
        next direct choice could take the slot a slower warp was claiming, and two
        positions ended on one row (about one position per such round, not in every
        run)."""
        helpers = _swap_helpers()
        hot, scratch, seq_len, window = 16384, 1024, 65536, 24000
        state = helpers._make_state(
            num_reqs=2,
            hot_buffer_size=hot,
            page_size=self.PAGE,
            scratch_size=scratch,
            seq_len=seq_len,
            item_words=1,
            metadata_occurrences=NUM_STEPS * NUM_TOP_K,
        )
        generator = torch.Generator().manual_seed(4)
        seq_lens = torch.full((2 * NUM_STEPS,), seq_len, dtype=torch.int32, device="cuda")
        # `core` positions are shared by the eight steps: the union is
        # core + 8 x (2048 - core) distinct positions.
        for round_idx, core in enumerate((1900, 256, 0, 128, 0, 64, 0, 0, 256, 0)):
            rows = []
            for req in range(2):
                low = (round_idx * 1500 + req * 700) % (seq_len - window)
                perm = low + torch.randperm(window, generator=generator)
                shared, rest = perm[:core], perm[core:]
                own = NUM_TOP_K - core
                rows.append(
                    torch.stack(
                        [
                            torch.cat((shared, rest[step * own : (step + 1) * own]))
                            for step in range(NUM_STEPS)
                        ]
                    )
                )
            top_k_tokens = torch.stack(rows).to(torch.int32).cuda().contiguous()
            with self.subTest(round=round_idx, core=core):
                out = helpers._run_swap(
                    top_k_tokens=top_k_tokens, seq_lens=seq_lens, state=state
                )
                torch.cuda.synchronize()
                self.assertTrue(torch.all(out >= 0).item())
                helpers._assert_output_matches_tokens(state, out, top_k_tokens)

    def test_zero_length_request_next_to_a_swapping_one(self):
        """A resident request reaches the swap with length zero in all eight steps:
        nothing is selected, copied or rotated for it, while the buffered request next
        to it resolves its misses."""
        helpers = _swap_helpers()
        state = self._state(helpers, 2, 32768)
        step = torch.arange(NUM_TOP_K, dtype=torch.int32, device="cuda") + self.HOT - 1548
        top_k_tokens = step.view(1, 1, -1).repeat(2, NUM_STEPS, 1).contiguous()
        seq_lens = torch.full((2 * NUM_STEPS,), 32768, dtype=torch.int32, device="cuda")
        seq_lens[:NUM_STEPS] = 0
        tokens_before = state.device_buffer_tokens.clone()
        locs_before = state.device_buffer_locs.clone()
        scratch_before = state.swap_state.scratch_locs.clone()
        index_before = state.swap_state.cache_index.clone()
        rows_per_req = state.device_buffer.shape[0] // 2
        rows_before = state.device_buffer[:rows_per_req].clone()

        out = helpers._run_swap(top_k_tokens=top_k_tokens, seq_lens=seq_lens, state=state)
        torch.cuda.synchronize()

        self.assertTrue(torch.all(out[0] == -1).item())
        torch.testing.assert_close(state.device_buffer_tokens[0], tokens_before[0])
        torch.testing.assert_close(state.device_buffer_locs[0], locs_before[0])
        torch.testing.assert_close(state.swap_state.scratch_locs[0], scratch_before[0])
        torch.testing.assert_close(state.swap_state.cache_index[0], index_before[0])
        torch.testing.assert_close(state.device_buffer[:rows_per_req], rows_before)
        helpers._assert_output_matches_tokens(state, out[1], top_k_tokens[1])


if __name__ == "__main__":
    unittest.main()
