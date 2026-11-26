"""HiSparseShardView: the row arithmetic of a context split over several GPUs.

A request table holds widened slots (slot s is row s // count of shard s % count); the paged
HiSparse tables are per shard and in rows. Without a split every method is the identity."""

import unittest

import torch

from sglang.srt.mem_cache.hisparse_shard import HiSparseShardView
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

P = 4  # rows per page


def _request_table(*, count: int, pages: list[int], num_tokens: int) -> torch.Tensor:
    """One request whose logical pages are `pages`: position p sits in page p // span at
    offset p % span, as the widened allocator hands slots out."""
    span = P * count
    table = torch.full((1, len(pages) * span), -1, dtype=torch.int32)
    for p in range(num_tokens):
        table[0, p] = pages[p // span] * span + p % span
    return table


class TestHiSparseShardView(CustomTestCase):
    def test_no_split_is_the_identity(self):
        shard = HiSparseShardView()
        self.assertFalse(shard.split)
        slots = torch.tensor([5, 6, 9])
        self.assertIs(shard.row_of(slots), slots)
        self.assertIs(shard.anchor_rows(slots), slots)
        self.assertTrue(bool(shard.is_anchor(slots).all()))
        for n in (0, 1, 63, 64, 65):
            self.assertEqual(shard.extent(n), n)
            self.assertEqual(shard.whole(n), n)
            self.assertEqual(shard.tokens(n), n)
        table = _request_table(count=1, pages=[7, 2, 5], num_tokens=10)
        torch.testing.assert_close(shard.rows(table, 0, 2, 9, 3), table[0, 2:9:3])

    def test_extent_counts_the_fullest_shard_and_whole_the_emptiest(self):
        shard = HiSparseShardView(count=4, rank=2)
        self.assertTrue(shard.split)
        for num_tokens, extent, whole in ((0, 0, 0), (1, 1, 0), (4, 1, 1), (5, 2, 1), (253, 64, 63)):
            self.assertEqual(shard.extent(num_tokens), extent)
            self.assertEqual(shard.whole(num_tokens), whole)
        self.assertEqual(shard.tokens(64), 256)
        lens = torch.tensor([1, 4, 5, 253])
        self.assertEqual(shard.extent(lens).tolist(), [1, 1, 2, 64])
        self.assertEqual(shard.whole(lens).tolist(), [0, 1, 1, 63])

    def test_rows_of_a_request_are_the_same_on_every_shard(self):
        count, pages, num_tokens = 4, [7, 2, 5], 37
        table = _request_table(count=count, pages=pages, num_tokens=num_tokens)
        extent = HiSparseShardView(count=count).extent(num_tokens)
        want = torch.tensor([pages[q // P] * P + q % P for q in range(extent)])
        for rank in range(count):
            shard = HiSparseShardView(count=count, rank=rank)
            got = shard.rows(table, 0, 0, extent).to(torch.int64)
            torch.testing.assert_close(got, want)
            # A page's first row, as the paging tables take it.
            first = shard.rows(table, 0, 0, extent // P * P, P).to(torch.int64)
            torch.testing.assert_close(first // P, torch.tensor(pages[: extent // P]))
        # The row a shard writes for its own position is the row the view names.
        for p in range(num_tokens):
            self.assertEqual(int(table[0, p]) // count, int(want[p // count]))
            self.assertEqual(int(table[0, p]) % count, p % count)

    def test_a_row_follows_its_anchor_slot(self):
        shard = HiSparseShardView(count=4, rank=1)
        slots = torch.tensor([40, 41, 42, 43, 44, 45, 51])
        self.assertEqual(shard.row_of(slots).tolist(), [10, 10, 10, 10, 11, 11, 12])
        self.assertEqual(shard.anchor_rows(slots).tolist(), [10, 11])
        self.assertEqual(
            shard.is_anchor(slots).tolist(), [True, False, False, False, True, False, False]
        )
        # Allocating then freeing any run of slots opens and closes the same rows.
        table = _request_table(count=4, pages=[3, 9], num_tokens=29)
        allocated = table[0, :29].to(torch.int64)
        opened = shard.anchor_rows(allocated)
        self.assertEqual(opened.numel(), shard.extent(29))
        tail = allocated[13:]
        kept = shard.anchor_rows(allocated[:13])
        self.assertEqual(
            sorted(kept.tolist() + shard.anchor_rows(tail).tolist()), sorted(opened.tolist())
        )


if __name__ == "__main__":
    unittest.main()
