"""Slot tables of HiSparse resident requests against gather references."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

NUM_REQUEST_SLOTS = 6
MAX_CONTEXT_LEN = 700
NUM_TOP_K = 300


def _request_tables():
    """Request slots 1 and 4 are resident; every request's positions map to distinct rows."""
    generator = torch.Generator().manual_seed(0)
    num_logical = NUM_REQUEST_SLOTS * MAX_CONTEXT_LEN
    req_to_token = torch.randperm(num_logical, generator=generator, dtype=torch.int32)
    req_to_token = req_to_token.view(NUM_REQUEST_SLOTS, MAX_CONTEXT_LEN).cuda()
    mapping = torch.randperm(num_logical, generator=generator) + 1000
    resident = torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.int8, device="cuda")
    return req_to_token, mapping.cuda(), resident


class TestHiSparseResidentRows(CustomTestCase):
    def test_only_resident_queries_are_rewritten(self):
        """Buffered rows hold the swap-in kernel's output and padded rows its -1 fill;
        a translation that touched them would hand attention another request's rows."""
        from sglang.kernels.ops.kvcache.hisparse_resident import translate_resident_rows

        req_to_token, mapping, resident = _request_tables()
        req_pool_indices = torch.tensor([4, 0, 1, 2, 4], device="cuda")
        for num_steps in (1, 3):
            with self.subTest(num_steps=num_steps):
                num_reqs = req_pool_indices.numel()
                shape = (
                    (num_reqs, NUM_TOP_K)
                    if num_steps == 1
                    else (num_reqs, num_steps, NUM_TOP_K)
                )
                top_k = torch.randint(-2, MAX_CONTEXT_LEN, shape, dtype=torch.int32)
                top_k = top_k.cuda()
                # Each step of a request sees a different length, as in a verify batch.
                seq_lens = torch.randint(
                    1, MAX_CONTEXT_LEN, (num_reqs * num_steps,), dtype=torch.int32
                ).cuda()
                output = torch.full(shape, -7, dtype=torch.int32, device="cuda")
                # The last batch row is CUDA-graph padding although its slot is resident.
                num_real_reqs = torch.tensor(
                    [num_reqs - 1], dtype=torch.int32, device="cuda"
                )

                translate_resident_rows(
                    top_k=top_k,
                    req_pool_indices=req_pool_indices,
                    seq_lens=seq_lens,
                    resident=resident,
                    req_to_token=req_to_token,
                    mapping=mapping,
                    num_real_reqs=num_real_reqs,
                    output=output,
                )

                positions = top_k.view(num_reqs, num_steps, NUM_TOP_K).long()
                lengths = seq_lens.view(num_reqs, num_steps, 1).long()
                selected = (positions >= 0) & (positions < lengths)
                logical = req_to_token[
                    req_pool_indices.view(-1, 1, 1), positions.clamp(min=0)
                ].long()
                rewritten = (
                    resident[req_pool_indices].bool()
                    & (torch.arange(num_reqs, device="cuda") < num_reqs - 1)
                ).view(-1, 1, 1)
                expected = torch.where(
                    rewritten,
                    torch.where(selected, mapping[logical], -1),
                    -7,
                ).to(torch.int32)
                torch.testing.assert_close(
                    output.view(num_reqs, num_steps, NUM_TOP_K),
                    expected,
                    rtol=0,
                    atol=0,
                )

    def test_masked_lengths_follow_the_request_of_each_query(self):
        """Under speculative decoding one request owns num_steps consecutive queries; a
        mask indexed per query instead would leave resident queries with their length and
        the swap-in kernel would evict into a hot buffer the request does not have."""
        from sglang.kernels.ops.kvcache.hisparse_resident import mask_resident_seq_lens

        _, _, resident = _request_tables()
        for num_reqs in (1, 5, 130):
            req_pool_indices = torch.randint(0, NUM_REQUEST_SLOTS, (num_reqs,)).cuda()
            for num_steps in (1, 3):
                for dtype in (torch.int32, torch.int64):
                    with self.subTest(
                        num_reqs=num_reqs, num_steps=num_steps, dtype=dtype
                    ):
                        seq_lens = torch.randint(
                            1, 1000, (num_reqs * num_steps,), dtype=dtype
                        ).cuda()
                        original = seq_lens.clone()

                        masked = mask_resident_seq_lens(
                            seq_lens=seq_lens,
                            req_pool_indices=req_pool_indices,
                            resident=resident,
                        )

                        is_resident = resident[req_pool_indices].bool()
                        expected = torch.where(
                            is_resident.repeat_interleave(num_steps), 0, original
                        )
                        torch.testing.assert_close(masked, expected, rtol=0, atol=0)
                        torch.testing.assert_close(seq_lens, original, rtol=0, atol=0)

    def test_page_table_covers_the_last_partial_page(self):
        """The top-k kernel reads the page of every position below the length, the
        partly filled last page included; entries past it may stay uninitialized."""
        from sglang.kernels.ops.kvcache.hisparse_resident import resident_page_table

        page_size = 64
        num_pages = 1100
        generator = torch.Generator().manual_seed(1)
        page_table = torch.randperm(4 * num_pages, generator=generator).view(4, -1)
        page_table = page_table.to(torch.int32).cuda()
        # Every logical page is mirrored by one whole page of rows.
        row_pages = torch.randperm(4 * num_pages, generator=generator).cuda() + 3
        mapping = (
            row_pages.view(-1, 1) * page_size + torch.arange(page_size, device="cuda")
        ).view(-1)
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                seq_lens = torch.tensor(
                    [1, page_size, page_size + 1, num_pages * page_size - 5],
                    dtype=dtype,
                    device="cuda",
                )

                actual = resident_page_table(
                    page_table=page_table,
                    mapping=mapping,
                    seq_lens=seq_lens,
                    page_size=page_size,
                )

                for row, length in enumerate(seq_lens.tolist()):
                    used = -(-length // page_size)
                    torch.testing.assert_close(
                        actual[row, :used].long(),
                        row_pages[page_table[row, :used].long()],
                        rtol=0,
                        atol=0,
                    )


if __name__ == "__main__":
    unittest.main()
