"""Paged HiSparse under a split context (SGLANG_HISPARSE_PAGED_DCP): the allocator hands out widened
slots and mirrors each GPU's rows page by page, the pool translates widened slots to widened slots
of its own rows, a decode step opens a row for the requests whose new position is shard 0's, and
the ledger's admission numbers depend on nothing a single GPU's pending work can move."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.mem_cache import hisparse_shard
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.hisparse_ledger import HiSparsePageLedger
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.mem_cache.hisparse_shard import HiSparseShardView
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

P = 4  # rows per page
N = 4  # shards
SPAN = P * N
DEVICE_ROWS = 16 * P


def _allocator(shard: HiSparseShardView) -> HiSparseTokenToKVPoolAllocator:
    pool = object.__new__(HiSparseDSATokenToKVPool)
    pool._hs_el = {"k_max": DEVICE_ROWS, "pinned": True}
    pool.hs_shard = shard
    pool.full_to_hisparse_device_index_mapping = None

    def limits(allocator):
        allocator._hs_el.update(dev_max_page=1 << 30, log_max_page=1 << 30)

    with patch.object(HiSparseTokenToKVPoolAllocator, "_hs_el_init_limits", limits):
        allocator = HiSparseTokenToKVPoolAllocator(
            size=DEVICE_ROWS,
            page_size=P,
            dtype=torch.float8_e4m3fn,
            device="cpu",
            kvcache=pool,
            need_sort=False,
            host_to_device_ratio=2,
            shard=shard,
        )
    # CPU tensors: the torch paths of the two paged allocators.
    allocator.logical_attn_allocator.use_triton_kernels = False
    allocator.hisparse_attn_allocator.use_triton_kernels = False
    return allocator


def _extend(allocator, start: int, end: int, last_loc: int) -> torch.Tensor:
    return allocator.alloc_extend(
        prefix_lens=torch.tensor([start]),
        prefix_lens_cpu=torch.tensor([start]),
        seq_lens=torch.tensor([end]),
        seq_lens_cpu=torch.tensor([end]),
        last_loc=torch.tensor([last_loc]),
        extend_num_tokens=end - start,
    )


class TestPagedDcpAllocator(CustomTestCase):
    def test_sizes_count_tokens_outside_and_rows_inside(self):
        allocator = _allocator(HiSparseShardView(count=N, rank=1))
        self.assertEqual(allocator.page_size, SPAN)  # what the scheduler rounds by
        self.assertEqual(allocator.row_page_size, P)
        self.assertEqual(allocator.rows_full, 2 * DEVICE_ROWS)
        self.assertEqual(allocator.size_full, 2 * DEVICE_ROWS * N)
        self.assertEqual(allocator.logical_attn_allocator.page_size, SPAN)
        self.assertEqual(allocator.hisparse_attn_allocator.page_size, P)
        self.assertEqual(
            allocator.full_to_hisparse_device_index_mapping.numel(),
            2 * DEVICE_ROWS + P + 1,
        )

    def test_no_split_keeps_one_unit(self):
        allocator = _allocator(HiSparseShardView())
        self.assertEqual(allocator.page_size, P)
        self.assertEqual(allocator.row_page_size, P)
        self.assertEqual(allocator.size_full, allocator.rows_full)

    def test_a_pool_built_for_another_shard_is_refused(self):
        pool = object.__new__(HiSparseDSATokenToKVPool)
        pool._hs_el = {"k_max": DEVICE_ROWS, "pinned": True}
        pool.hs_shard = HiSparseShardView(count=N, rank=0)
        with self.assertRaisesRegex(ValueError, "same shard"):
            HiSparseTokenToKVPoolAllocator(
                size=DEVICE_ROWS,
                page_size=P,
                dtype=torch.float8_e4m3fn,
                device="cpu",
                kvcache=pool,
                need_sort=False,
                shard=HiSparseShardView(count=N, rank=1),
            )

    def test_rows_mirror_the_logical_pages_on_every_shard(self):
        num_tokens = 2 * SPAN + 5  # the prompt ends inside a row of its third page
        placed = []
        for rank in range(N):
            shard = HiSparseShardView(count=N, rank=rank)
            allocator = _allocator(shard)
            free_rows = allocator.hisparse_attn_allocator.available_size()
            first = _extend(allocator, 0, 7, -1)
            rest = _extend(allocator, 7, num_tokens, int(first[-1]))
            slots = torch.cat([first, rest])
            # Widened slots: whole pages of SPAN ids, positions in order inside a page.
            self.assertTrue(bool((slots % SPAN == torch.arange(num_tokens) % SPAN).all()))
            mapping = allocator.full_to_hisparse_device_index_mapping
            extent = shard.extent(num_tokens)
            rows = slots[::N] // N  # the row of every N-th position: the same on every shard
            self.assertEqual(rows.numel(), extent)
            device_rows = mapping[rows]
            self.assertTrue(bool((device_rows > 0).all()))
            # Page for page: row q of a request sits at offset q % P of its page's frame.
            self.assertTrue(bool((device_rows % P == torch.arange(extent) % P).all()))
            self.assertEqual(int((mapping[:-1] > 0).sum()), extent)
            # The row this shard writes for each of its own positions is mapped.
            own = slots[rank::N]
            self.assertTrue(bool((mapping[own // N] > 0).all()))
            placed.append((slots.clone(), extent))
            allocator.free(slots)
            self.assertEqual(int((mapping[:-1] > 0).sum()), 0)
            self.assertEqual(
                allocator.hisparse_attn_allocator.available_size(), free_rows
            )
            self.assertEqual(
                allocator.logical_attn_allocator.available_size(), allocator.size_full
            )
        # Every shard was handed the same logical slots and the same number of rows.
        for slots, extent in placed[1:]:
            torch.testing.assert_close(slots, placed[0][0])
            self.assertEqual(extent, placed[0][1])

    def test_a_chunk_that_opens_no_row_takes_no_device_row(self):
        allocator = _allocator(HiSparseShardView(count=N, rank=2))
        first = _extend(allocator, 0, 5, -1)  # rows 0 and 1
        before = allocator.hisparse_attn_allocator.available_size()
        mapped = int((allocator.full_to_hisparse_device_index_mapping[:-1] > 0).sum())
        more = _extend(allocator, 5, 8, int(first[-1]))  # positions 5..7 are in row 1
        self.assertEqual(more.numel(), 3)
        self.assertEqual(allocator.hisparse_attn_allocator.available_size(), before)
        self.assertEqual(
            int((allocator.full_to_hisparse_device_index_mapping[:-1] > 0).sum()), mapped
        )


class TestPagedDcpTranslation(CustomTestCase):
    def test_page_tables_and_cpu_inputs_keep_shard_and_padding(self):
        pool = object.__new__(HiSparseDSATokenToKVPool)
        pool.hs_shard = HiSparseShardView(count=N, rank=3)
        mapping = torch.cat([torch.arange(40) + 100, torch.tensor([-1])])
        pool.full_to_hisparse_device_index_mapping = mapping
        slots = torch.tensor([[17, 18, -1], [39 * N + 3, 0, -1]])
        out = pool.translate_loc_to_hisparse_device(slots)
        expected = torch.tensor(
            [[104 * N + 1, 104 * N + 2, -1], [139 * N + 3, 100 * N, -1]]
        )
        torch.testing.assert_close(out, expected)

    def test_a_split_pool_needs_the_paged_switch(self):
        pool = object.__new__(HiSparseDSATokenToKVPool)
        with patch(
            "sglang.srt.mem_cache.hisparse_memory_pool.envs.SGLANG_HISPARSE_PAGED.get",
            return_value=False,
        ):
            with self.assertRaisesRegex(ValueError, "SGLANG_HISPARSE_PAGED"):
                HiSparseDSATokenToKVPool.__init__(
                    pool,
                    size=64,
                    page_size=P,
                    kv_lora_rank=8,
                    dtype=torch.float8_e4m3fn,
                    qk_rope_head_dim=4,
                    layer_num=1,
                    device="cpu",
                    index_head_dim=4,
                    enable_memory_saver=False,
                    kv_cache_dim=12,
                    shard=HiSparseShardView(count=N, rank=0),
                )

    def test_the_worker_view_follows_the_switch_and_the_split(self):
        parallel = SimpleNamespace(attn_dcp_size=4, attn_dcp_rank=2)
        with patch("sglang.srt.runtime_context.get_parallel", return_value=parallel):
            with patch.object(
                hisparse_shard.envs.SGLANG_HISPARSE_PAGED_DCP, "get", return_value=True
            ):
                self.assertEqual(
                    hisparse_shard.current_shard_view(),
                    HiSparseShardView(count=4, rank=2),
                )
            with patch.object(
                hisparse_shard.envs.SGLANG_HISPARSE_PAGED_DCP, "get", return_value=False
            ):
                self.assertFalse(hisparse_shard.current_shard_view().split)
            parallel.attn_dcp_size = 1
            with patch.object(
                hisparse_shard.envs.SGLANG_HISPARSE_PAGED_DCP, "get", return_value=True
            ):
                self.assertFalse(hisparse_shard.current_shard_view().split)


class TestPagedDcpDecodeStep(CustomTestCase):
    def test_a_row_opens_with_shard_zeros_position_on_every_shard(self):
        for rank in (0, 3):
            shard = HiSparseShardView(count=N, rank=rank)
            allocator = _allocator(shard)
            req_to_token = torch.full((2, 8 * SPAN), -1, dtype=torch.int32)
            lens = [SPAN, 9]  # request 0 ends on a page, request 1 inside row 2
            last = []
            for rid, num_tokens in enumerate(lens):
                slots = _extend(allocator, 0, num_tokens, -1)
                req_to_token[rid, :num_tokens] = slots.to(torch.int32)
                last.append(int(slots[-1]))
            coordinator = object.__new__(HiSparseCoordinator)
            coordinator.shard = shard
            coordinator.device = "cpu"
            coordinator.mem_pool_device = SimpleNamespace(page_size=P)
            coordinator.token_to_kv_pool_allocator = allocator
            coordinator.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
            coordinator._hs_lean = [0, 0]
            coordinator._resident_steps = 0
            coordinator.reclaim_device = MagicMock(return_value=False)
            coordinator._gpu_rows = lambda rows: torch.tensor(rows, dtype=torch.int64)
            mapping = allocator.full_to_hisparse_device_index_mapping
            mapped = int((mapping[:-1] > 0).sum())
            free = allocator.hisparse_attn_allocator.available_size()
            # Step 1: request 0's new position SPAN is shard 0's and opens a page;
            # request 1's new position 9 lies in a row that exists.
            seq = torch.tensor([lens[0] + 1, lens[1] + 1])
            out = allocator.alloc_decode(seq, seq, torch.tensor(last))
            req_to_token[torch.arange(2), seq - 1] = out.to(torch.int32)
            coordinator._hs_split_decode(seq, out, torch.arange(2), seq)
            self.assertEqual(int((mapping[:-1] > 0).sum()), mapped + 1)
            self.assertEqual(
                allocator.hisparse_attn_allocator.available_size(), free - P
            )
            new_row = int(mapping[int(out[0]) // N])
            self.assertEqual(new_row % P, 0)
            # Every position of both requests names a mapped row of this shard.
            for rid in range(2):
                slots = req_to_token[rid, : int(seq[rid])].to(torch.int64)
                self.assertTrue(bool((mapping[slots // N] > 0).all()))
            coordinator.reclaim_device.assert_not_called()


def _ledger(*, logical_pages=64, device_pages=16, host_pages=24):
    logical = SimpleNamespace(available_size=lambda: logical.free_pages * SPAN)
    logical.free_pages = logical_pages
    device = SimpleNamespace(available_size=lambda: device_pages * P)
    host = SimpleNamespace(available_size=lambda: host_pages * P)
    allocator = SimpleNamespace(
        row_page_size=P,
        page_size=SPAN,
        logical_attn_allocator=logical,
        hisparse_attn_allocator=device,
    )
    coordinator = SimpleNamespace(
        token_to_kv_pool_allocator=allocator,
        mem_pool_host=host,
        _resident={},
        paging=SimpleNamespace(sealable_end=lambda req: req.sealable_rows),
    )
    return HiSparsePageLedger(coordinator), coordinator, logical


def _request(num_tokens: int, sealable_rows: int = 0):
    return SimpleNamespace(
        kv=SimpleNamespace(kv_allocated_len=num_tokens), sealable_rows=sealable_rows
    )


class TestPagedDcpLedger(CustomTestCase):
    def test_the_empty_pool_offers_its_device_frames(self):
        ledger, _, _ = _ledger()
        self.assertEqual(ledger.available_pages(), 16)
        self.assertEqual(ledger.available_tokens(), 16 * SPAN)
        self.assertFalse(ledger.holds_request_tokens())

    def test_pinned_pages_limit_until_they_become_movable(self):
        ledger, coordinator, logical = _ledger()
        logical.free_pages -= 12  # a prefilling request of 12 pages: all pinned
        self.assertEqual(ledger.counts(), (12, 0))
        self.assertEqual(ledger.available_pages(), 4)
        # Resident, 11 whole pages below its writing position: they may go to host.
        coordinator._resident[5] = _request(12 * SPAN, sealable_rows=11 * P)
        self.assertEqual(ledger.counts(), (12, 11))
        self.assertEqual(ledger.available_pages(), 15)

    def test_every_page_needs_a_place(self):
        ledger, coordinator, logical = _ledger(logical_pages=64, device_pages=16, host_pages=24)
        logical.free_pages -= 36
        coordinator._resident[1] = _request(36 * SPAN, sealable_rows=35 * P)
        # 36 live of 40 places: 4 more pages, although only one is pinned.
        self.assertEqual(ledger.available_pages(), 4)

    def test_the_logical_ids_limit(self):
        ledger, coordinator, logical = _ledger(logical_pages=20, device_pages=16, host_pages=24)
        logical.free_pages -= 18
        coordinator._resident[1] = _request(18 * SPAN, sealable_rows=18 * P)
        self.assertEqual(ledger.available_pages(), 2)

    def test_digests_agree_only_on_equal_ledgers_and_equal_answers(self):
        a, _, logical_a = _ledger()
        b, _, logical_b = _ledger()
        self.assertEqual(a.digest(), b.digest())
        a.available_tokens()  # how often a scheduler asks does not matter
        self.assertEqual(a.digest(), b.digest())
        logical_b.free_pages -= 1
        self.assertNotEqual(a.digest(), b.digest())
        c, _, _ = _ledger(host_pages=25)
        self.assertNotEqual(a.digest(), c.digest())

    def test_the_audit_raises_on_every_rank_when_one_differs(self):
        coordinator = object.__new__(HiSparseCoordinator)
        coordinator.ledger, _, _ = _ledger()
        coordinator._audit_every = 2
        coordinator._audit_steps = 0
        coordinator._group_min_max = lambda value: (value, value)
        for _ in range(4):
            coordinator._lockstep_tick()
        coordinator._group_min_max = lambda value: (value - 1, value)
        coordinator._lockstep_tick()  # not an audit step
        with self.assertRaisesRegex(RuntimeError, "no longer agree"):
            coordinator._lockstep_tick()

    def test_one_rank_reduces_to_itself(self):
        coordinator = object.__new__(HiSparseCoordinator)
        coordinator.tp_world_size = 1
        self.assertEqual(coordinator._group_min_max(12345), (12345, 12345))


if __name__ == "__main__":
    unittest.main()
