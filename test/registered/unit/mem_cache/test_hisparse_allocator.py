import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestHiSparseDecodeRemap(CustomTestCase):
    def test_spec_scratch_is_charged_only_before_request_slot_allocation(self):
        allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
        allocator.page_size = 64

        allocator.configure_spec_scratch(1024)

        self.assertEqual(allocator.request_slot_reserve(has_req_pool_slot=False), 1024)
        self.assertEqual(allocator.request_slot_reserve(has_req_pool_slot=True), 0)

    def test_buffered_prompt_host_rows_are_charged_under_exclusive_placement(self):
        """A prompt that stays buffered holds host slots besides its device slots."""
        allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
        allocator.page_size = 64
        allocator.configure_spec_scratch(0)

        def reserve(prefill_tokens, has_req_pool_slot=False):
            return allocator.request_slot_reserve(
                has_req_pool_slot=has_req_pool_slot, prefill_tokens=prefill_tokens
            )

        self.assertEqual(reserve(1000), 0)
        allocator._buffered_prompt_tokens = 12288
        self.assertEqual(reserve(1000), 1024)
        self.assertEqual(reserve(20000), 0)
        # It took its host rows with its request slot.
        self.assertEqual(reserve(1000, has_req_pool_slot=True), 0)

    def test_buffer_extra_is_the_buffer_charge_of_the_slot_reserve_taken_at_once(self):
        """The slots a short prompt's buffer adds to its prefill pages: room is
        made by eviction if the free list is short; a prompt whose pages cover
        the buffer takes none."""
        allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
        allocator.page_size = 64
        allocator.device = "cpu"
        allocator._hs_el = None
        allocator.configure_spec_scratch(1024, device_buffer_size=12352)
        free = [640]
        reclaimed = []

        def alloc(need):
            if need > free[0]:
                return None
            free[0] -= need
            return torch.arange(64, 64 + need)

        def reclaim_device(need):
            reclaimed.append(need)
            free[0] = max(free[0], need)

        allocator.hisparse_attn_allocator = SimpleNamespace(
            available_size=lambda: free[0], alloc=alloc
        )
        allocator._reclaimer = SimpleNamespace(reclaim_device=reclaim_device)

        self.assertEqual(allocator.alloc_buffer_extra(20000).numel(), 0)
        self.assertEqual((free[0], reclaimed), (640, []))
        # 12352 - 2048: what request_slot_reserve charges besides the scratch.
        self.assertEqual(
            allocator.request_slot_reserve(has_req_pool_slot=True, prefill_tokens=2000),
            10304,
        )
        self.assertEqual(allocator.alloc_buffer_extra(2000).numel(), 10304)
        self.assertEqual((free[0], reclaimed), (0, [10304]))
        # Nothing can be evicted for it: the caller is told, nothing is taken.
        allocator._reclaimer = SimpleNamespace(reclaim_device=lambda need: None)
        free[0] = 128
        self.assertIsNone(allocator.alloc_buffer_extra(2000))
        self.assertEqual(free[0], 128)

    def test_pd_transfer_keeps_its_buffer_slots_until_the_buffer_is_built(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = object.__new__(HiSparseCoordinator)
        slots = torch.arange(64, 128)
        freed = []
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            alloc_buffer_extra=MagicMock(
                side_effect=[slots, torch.empty(0, dtype=torch.int64), None]
            ),
            free_buffer_extra=freed.append,
        )
        short, resident, refused = (
            SimpleNamespace(kv=SimpleNamespace(req_pool_idx=idx)) for idx in (3, 4, 5)
        )

        self.assertTrue(coordinator.hold_buffer_slots(short, 2000))
        self.assertTrue(coordinator.hold_buffer_slots(resident, 20000))
        self.assertFalse(coordinator.hold_buffer_slots(refused, 2000))
        self.assertEqual(list(coordinator._recv_buffer_hold), [3])

        coordinator._give_back_buffer_hold(4)
        coordinator._give_back_buffer_hold(5)
        self.assertEqual(freed, [])
        coordinator._give_back_buffer_hold(3)
        coordinator._give_back_buffer_hold(3)
        self.assertEqual(len(freed), 1)
        self.assertIs(freed[0], slots)
        # A coordinator that never kept slots (also one built without __init__).
        object.__new__(HiSparseCoordinator)._give_back_buffer_hold(0)

    def test_a_prealloc_without_frames_gives_back_what_it_took_with_its_row(self):
        """The reclaim behind the room check can come back short. The request then
        holds its scratch, a short prompt's host rows and its row with nothing in
        it: a later release would free the row's previous occupant."""
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = object.__new__(HiSparseCoordinator)
        freed = []
        coordinator._recv_buffer_hold = {3: torch.arange(128, 192)}
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            free_buffer_extra=lambda slots: freed.append(("buffer", slots.tolist()[:1]))
        )
        coordinator.spec_swap = SimpleNamespace(
            free_unrotated_scratch=lambda rid: freed.append(("scratch", rid))
        )
        coordinator.req_to_host_pool = torch.full((4, 8), -1, dtype=torch.int64)
        coordinator.req_to_host_pool[3, :4] = torch.arange(40, 44)
        coordinator.req_to_host_pool_allocated_len = torch.tensor([0, 0, 0, 4])
        coordinator.mem_pool_host = SimpleNamespace(
            allocated_host_indices=lambda table, rid, n: table[rid, : int(n)],
            free=lambda slots: freed.append(("host", slots.tolist())),
        )
        req = SimpleNamespace(kv=ReqKvInfo())
        req.kv.req_pool_idx = 3

        coordinator.cancel_prealloc(req)

        self.assertEqual(
            freed, [("buffer", [128]), ("scratch", 3), ("host", [40, 41, 42, 43])]
        )
        self.assertEqual(coordinator._recv_buffer_hold, {})
        self.assertTrue(bool((coordinator.req_to_host_pool[3] == -1).all()))
        self.assertEqual(int(coordinator.req_to_host_pool_allocated_len[3]), 0)

    def test_device_pages_are_freed_in_ascending_order(self):
        # An eviction hands its frames over in the order it picked them; the
        # free list gets them as free_hisparse_indices would return them.
        freed = []
        allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
        allocator.hisparse_attn_allocator = SimpleNamespace(
            free_page_ids=lambda pages: freed.append(pages.tolist())
        )
        allocator.free_device_pages(torch.tensor([9, 3, 7, 4]))
        self.assertEqual(freed, [[3, 4, 7, 9]])

    def test_paged_pool_budget_marks_a_refusal_no_wait_can_lift(self):
        """With paged residency a request refused while no request holds logical
        slots is unschedulable; without it the allocator keeps the plain budget."""
        from sglang.srt.mem_cache.prefill_budget import (
            HiSparsePagedPrefillBudget,
            PrefillBudget,
        )

        logical_free = [6400]
        allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
        allocator.page_size = 64
        allocator._size_full = 6400
        allocator._hs_el = None
        allocator.logical_attn_allocator = SimpleNamespace(
            available_size=lambda: logical_free[0], num_pages=100
        )
        allocator.hisparse_attn_allocator = SimpleNamespace(available_size=lambda: 1280)
        tree_cache = SimpleNamespace(
            supports_mamba=lambda: False, evictable_size=lambda: 0
        )
        self.assertIs(type(allocator.create_prefill_budget(tree_cache)), PrefillBudget)

        spare_scratch = [0]
        allocator._reclaimer = SimpleNamespace(
            hs_exclusive=False,
            reclaimable_slots=lambda: 0,
            spare_scratch_slots=lambda: spare_scratch[0],
        )
        budget = allocator.create_prefill_budget(tree_cache)
        self.assertIsInstance(budget, HiSparsePagedPrefillBudget)

        def fits(paged_input):
            return budget.can_allocate_prefill(
                paged_input=paged_input,
                extend_input_len=paged_input,
                max_new_tokens=0,
                chunk_limit=None,
            )

        self.assertTrue(fits(1280))
        self.assertFalse(budget.unschedulable)
        # Larger than the device pool, and nothing will give memory back.
        self.assertFalse(fits(1344))
        self.assertTrue(budget.unschedulable)
        # A request holds slots: the refused one waits for it.
        logical_free[0] = 6336
        self.assertFalse(fits(1344))
        self.assertFalse(budget.unschedulable)
        # Scratch blocks kept for the next arrivals are free memory: an idle pool
        # that holds some still takes the request.
        logical_free[0] = 6400
        spare_scratch[0] = 64
        self.assertTrue(fits(1344))
        self.assertFalse(budget.unschedulable)
        # A request admitted earlier in the same pass will hold memory.
        logical_free[0] = 6400
        budget.total_offset = 64
        self.assertFalse(fits(1344))
        self.assertFalse(budget.unschedulable)

    def test_page_size_one_reclaims_temporary_device_slot(self):
        """Decode remapping must reclaim its temporary slot without freeing the live slot."""
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
        from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool

        pool = MiniMaxSparseKVPool(
            size=8,
            page_size=1,
            dtype=torch.float32,
            head_num=1,
            head_dim=8,
            idx_head_dim=16,
            dense_layer_ids=[0],
            sparse_layer_ids=[1],
            disable_value_sparse_layer_ids=[1],
            device="cpu",
            start_layer=0,
            end_layer=2,
            enable_hisparse=True,
        )
        allocator = HiSparseTokenToKVPoolAllocator(
            size=pool.size,
            page_size=1,
            dtype=pool.dtype,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.is_dsv4_hisparse = False
        coordinator.mem_pool_device = pool.main_pool
        coordinator.token_to_kv_pool_allocator = allocator
        coordinator.device_buffer_size = 2
        coordinator.req_to_device_buffer = allocator.hisparse_attn_allocator.alloc(
            3
        ).reshape(1, 3)
        coordinator.req_device_buffer_size = torch.tensor([3])
        coordinator.req_device_buffer_token_locs = torch.zeros(
            (1, 1, 3), dtype=torch.int32
        )
        coordinator._skip_first_backup = [True]
        out_loc = allocator.alloc(1)
        with patch("sglang.srt.managers.hisparse_coordinator._is_hip", False):
            for _ in range(2):
                coordinator._skip_first_backup[0] = True
                coordinator.map_last_loc_to_buffer(
                    seq_lens=torch.tensor([3]),
                    out_cache_loc=out_loc,
                    req_pool_indices=torch.tensor([0]),
                    seq_lens_cpu=torch.tensor([3]),
                    req_pool_indices_cpu=torch.tensor([0]),
                )
                self.assertEqual(
                    allocator.hisparse_attn_allocator.available_size(), pool.size - 3
                )
                torch.testing.assert_close(
                    allocator.full_to_hisparse_device_index_mapping[out_loc],
                    coordinator.req_to_device_buffer[:, 2],
                )


class TestDeepSeekV4HiSparseAllocator(CustomTestCase):
    def setUp(self):
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def test_free_returns_c4_slots(self):
        """Freeing a request's logical row must return the C4 HiSparse slots its
        compressed tokens were mapped to."""
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )

        page_size, full_size = 256, 4 * 256
        override = get_context().override_server_args(page_size=page_size)
        override.install()
        self.addCleanup(override.restore)
        pool = DeepSeekV4TokenToKVPool(
            max_num_reqs=4,
            swa_size=full_size,
            c4_size=full_size // 4,
            c128_size=2 * full_size // 128,  # sizes the C4 logical index space
            c4_state_pool_size=16,
            c128_state_pool_size=0,
            page_size=page_size,
            swa_page_size=page_size,
            dtype=torch.float8_e4m3fn,
            c4_state_dtype=torch.float32,
            c128_state_dtype=torch.float32,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            indexer_head_dim=128,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            compression_ratios=[0, 4],
            full_size=full_size,
            enable_hisparse=True,
        )
        logical = PagedTokenToKVPoolAllocator(
            full_size, page_size, pool.dtype, "cpu", pool, need_sort=False
        )
        allocator = DeepSeekV4HiSparseTokenToKVPoolAllocator(logical)
        c4_free = allocator.hisparse_attn_allocator.available_size()

        # What alloc_extend leaves behind for a one-page request.
        locs = logical.alloc(page_size)
        compressed = allocator.hisparse_kvcache.translate_loc_from_full_to_compressed(
            locs
        )
        slots = allocator.hisparse_attn_allocator.alloc(len(compressed))
        allocator.full_to_hisparse_device_index_mapping[compressed] = slots

        allocator.free(locs)

        self.assertEqual(allocator.hisparse_attn_allocator.available_size(), c4_free)
        self.assertFalse(
            allocator.full_to_hisparse_device_index_mapping[compressed].any()
        )

    def test_forwards_swa_tail_allocation_to_logical_allocator(self):
        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        logical_allocator = MagicMock(spec=["alloc_extend_swa_tail"])
        allocator.logical_attn_allocator = logical_allocator

        expected = torch.tensor([8, 9, 10], dtype=torch.int64)
        logical_allocator.alloc_extend_swa_tail.return_value = expected

        prefix_lens = torch.tensor([0], dtype=torch.int64)
        prefix_lens_cpu = torch.tensor([0], dtype=torch.int64)
        seq_lens = torch.tensor([512], dtype=torch.int64)
        seq_lens_cpu = torch.tensor([512], dtype=torch.int64)
        last_loc = torch.tensor([-1], dtype=torch.int64)

        result = allocator.alloc_extend_swa_tail(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=512,
            swa_tail_len=128,
        )

        self.assertIs(result, expected)
        logical_allocator.alloc_extend_swa_tail.assert_called_once()
        _, kwargs = logical_allocator.alloc_extend_swa_tail.call_args
        self.assertIs(kwargs["prefix_lens"], prefix_lens)
        self.assertIs(kwargs["prefix_lens_cpu"], prefix_lens_cpu)
        self.assertIs(kwargs["seq_lens"], seq_lens)
        self.assertIs(kwargs["seq_lens_cpu"], seq_lens_cpu)
        self.assertIs(kwargs["last_loc"], last_loc)
        self.assertEqual(kwargs["extend_num_tokens"], 512)
        self.assertEqual(kwargs["swa_tail_len"], 128)

    def test_forwards_prealloc_reclaim_to_logical_allocator(self):
        """PD decode preallocation must not crash on the HiSparse composite."""
        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        logical_allocator = MagicMock(spec=["reclaim_for_prealloc"])
        allocator.logical_attn_allocator = logical_allocator
        logical_allocator.reclaim_for_prealloc.return_value = None

        tree_cache = object()
        demand = KVDemand(512, {"swa": 256})
        self.assertIsNone(allocator.reclaim_for_prealloc(tree_cache, demand))
        logical_allocator.reclaim_for_prealloc.assert_called_once_with(
            tree_cache, demand
        )

        logical_allocator.reclaim_for_prealloc.return_value = "SWA eviction short"
        self.assertEqual(
            allocator.reclaim_for_prealloc(tree_cache, demand),
            "SWA eviction short",
        )

    def test_hisparse_budget_uses_full_logical_capacity_for_swa_tail(self):
        from sglang.srt.disaggregation.decode import DecodePreallocQueue

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        logical_allocator = SimpleNamespace(
            available_size=MagicMock(return_value=32),
            full_available_size=MagicMock(return_value=512),
        )
        queue.token_to_kv_pool_allocator = SimpleNamespace(
            logical_attn_allocator=logical_allocator,
            storage=None,
            pools_share_memory=False,
        )
        queue.scheduler = SimpleNamespace(enable_hisparse=True, last_batch=None)
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._need_space_for_single_req = MagicMock(return_value=0)
        queue._active_reserved_tokens = MagicMock(return_value=0)

        budget = queue._allocatable_token_budgets()

        self.assertEqual(budget, 512)
        logical_allocator.full_available_size.assert_called_once_with()
        logical_allocator.available_size.assert_not_called()

    def test_hisparse_prealloc_uses_swa_tail_for_direct_host_path(self):
        from sglang.srt.disaggregation.decode import DecodePreallocQueue

        fill_len = 512
        sliding_window_size = 200
        # The window tail floors the window start to a page boundary:
        # floor_align(512 - 200, 256) = 256, so the tail is one whole page.
        swa_tail_len = 256
        kv_loc = torch.arange(512, 512 + fill_len, dtype=torch.int64)
        host_indices = torch.arange(1000, 1128, dtype=torch.int64)

        req = SimpleNamespace(
            rid="req-0",
            origin_input_ids=list(range(fill_len)),
            output_ids=[],
            kv=ReqKvInfo(),
        )

        class ReqToTokenPool:
            def __init__(self):
                self.writes = []

            def alloc(self, reqs):
                for item in reqs:
                    item.kv.req_pool_idx = 0
                return torch.tensor([0], dtype=torch.int64)

            def write(self, indices, values):
                self.writes.append((indices, values))

        req_to_token_pool = ReqToTokenPool()
        allocator = SimpleNamespace(
            state=None,
            storage=None,
            pools_share_memory=False,
            # The window pool, keeping the scheduler's window.
            window_pool_names=lambda: ("swa",),
            target_window_pool=lambda: "swa",
            get_kvcache=lambda: SimpleNamespace(request_window=None),
            device=torch.device("cpu"),
            page_size=256,
            available_size=MagicMock(return_value=fill_len),
            swa_available_size=MagicMock(return_value=swa_tail_len),
            alloc_extend_swa_tail=MagicMock(return_value=kv_loc),
            alloc_logical_only=MagicMock(return_value=kv_loc),
        )
        # The window pool keeps the scheduler's window.
        kv_storage_of(req_to_token_pool, allocator).bind_window_retention(
            sliding_window_size, None
        )
        regular_host_alloc = MagicMock(return_value=host_indices)
        coordinator = SimpleNamespace(
            mem_pool_host=SimpleNamespace(alloc_paged_token_slots=regular_host_alloc),
            req_to_host_pool=object(),
            req_to_host_pool_allocated_len=object(),
            host_token_len=MagicMock(side_effect=lambda token_len: token_len // 4),
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.req_to_token_pool = req_to_token_pool
        queue.token_to_kv_pool_allocator = allocator
        queue.tree_cache = SimpleNamespace(
            evictable_size=MagicMock(return_value=0),
            protected_size=MagicMock(return_value=0),
            maybe_hand_to_session=lambda req: None,
        )
        queue.scheduler = SimpleNamespace(
            enable_hisparse=True,
            hisparse_coordinator=coordinator,
            server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=False),
            sliding_window_size=sliding_window_size,
        )
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)

        result = queue._pre_alloc(req)

        self.assertTrue(torch.equal(result, host_indices))
        allocator.alloc_extend_swa_tail.assert_called_once()
        allocator.alloc_logical_only.assert_not_called()
        _, kwargs = allocator.alloc_extend_swa_tail.call_args
        self.assertEqual(kwargs["extend_num_tokens"], fill_len)
        self.assertEqual(kwargs["swa_tail_len"], swa_tail_len)
        self.assertEqual(req.kv.get_evicted_seqlen("swa"), fill_len - swa_tail_len)
        self.assertEqual(req.kv.kv_allocated_len, fill_len)
        self.assertEqual(req.kv.kv_committed_len, fill_len)
        self.assertEqual(req.extend_end, fill_len)
        self.assertEqual(len(req_to_token_pool.writes), 1)
        coordinator.host_token_len.assert_called_once_with(fill_len)
        regular_host_alloc.assert_called_once_with(
            coordinator.req_to_host_pool,
            coordinator.req_to_host_pool_allocated_len,
            req.kv.req_pool_idx,
            0,
            len(host_indices),
        )
        self.assertTrue(torch.equal(req_to_token_pool.writes[0][1], kv_loc))

        # C4 indexer/C128 use the logical allocator's full-page IDs. They do not
        # use either the independently allocated host pages or C4 sparse slots.
        np.testing.assert_array_equal(
            np.unique(kv_loc.numpy() // allocator.page_size),
            np.array([2, 3]),
        )

    def test_mooncake_uses_separate_host_and_device_page_indices(self):
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

        manager = object.__new__(MooncakeKVManager)
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.enable_custom_mem_pool = False
        manager.max_transfer_batch_indices = 0
        manager._transfer_data = MagicMock(return_value=0)

        with ThreadPoolExecutor(max_workers=1) as executor:
            ret = manager._send_kvcache_generic(
                mooncake_session_id="session",
                src_data_ptrs=[1000, 2000, 3000],
                dst_data_ptrs=[10000, 20000, 30000],
                item_lens=[100, 100, 100],
                prefill_data_indices=np.array([1, 2], dtype=np.int32),
                dst_data_indices=np.array([7, 8], dtype=np.int32),
                executor=executor,
                dst_device_data_indices=np.array([21, 22], dtype=np.int32),
                dst_device_data_ptrs={20000, 30000},
            )

        self.assertEqual(ret, 0)
        manager._transfer_data.assert_called_once_with(
            "session",
            [
                (1100, 10700, 200),
                (2100, 22100, 200),
                (3100, 32100, 200),
            ],
        )

    def test_mooncake_derives_device_buffers_from_local_pp_layout(self):
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

        manager = object.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[1000, 2000, 3000],
            kv_item_lens=[100, 100, 100],
            kv_layer_ids=[],
            mla_compression_ratios=[4, 128, 4, 128],
            prefill_start_layer=0,
            prefill_end_layer=2,
        )
        manager._send_kvcache_generic = MagicMock(return_value=0)
        executor = MagicMock()

        # send_kvcache reads the memory bag (the unified-memory envelope-layout
        # check), so the context has to be published. This is the non-unified
        # path -- pin that explicitly rather than leaning on the default.
        with get_context().override_server_args(enable_unified_memory=False):
            manager.send_kvcache(
                "session",
                np.array([1], dtype=np.int32),
                [10000, 20000, 30000],
                np.array([7], dtype=np.int32),
                executor,
                dst_device_kv_indices=np.array([21], dtype=np.int32),
            )

        kwargs = manager._send_kvcache_generic.call_args.kwargs
        self.assertEqual(kwargs["dst_device_data_ptrs"], {20000, 30000})

    def test_mooncake_transfer_metadata_carries_device_page_indices(self):
        from sglang.srt.disaggregation.mooncake.conn import TransferInfo

        host_pages = np.array([7, 8], dtype=np.int32)
        device_pages = np.array([21, 22], dtype=np.int32)
        info = TransferInfo.from_zmq(
            [
                b"9",
                b"127.0.0.1",
                b"12345",
                b"session",
                host_pages.tobytes(),
                b"0",
                b"",
                b"1",
                b"0",
                device_pages.tobytes(),
            ]
        )

        np.testing.assert_array_equal(info.dst_kv_indices, host_pages)
        np.testing.assert_array_equal(info.dst_device_kv_indices, device_pages)


class TestHiSparseRetraction(CustomTestCase):
    def test_retraction_rebootstraps_instead_of_host_backup(self):
        from sglang.srt.mem_cache.common import retraction_needs_rebootstrap

        allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
        self.assertTrue(retraction_needs_rebootstrap(allocator))

    def _allocator(self, free, reclaimer):
        allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
        allocator.available_size = lambda: free[0]
        allocator.evict_to_free_tokens = lambda tree_cache, num_tokens: None
        allocator._reclaimer = reclaimer
        return allocator

    def test_a_decode_shortfall_takes_the_ended_requests_pages_first(self):
        """An ended request keeps its pages until its last forward has run; the
        step that would retract for want of them waits for that instead."""
        free = [0]
        drains = []

        def run_deferred_releases(block=False):
            drains.append(block)
            free[0] = 128

        allocator = self._allocator(
            free, SimpleNamespace(run_deferred_releases=run_deferred_releases)
        )
        self.assertTrue(allocator.check_decode_capacity(num_tokens=64, tree_cache=None))
        self.assertEqual(drains, [True])

    def test_a_step_that_fits_does_not_wait_for_the_ended_requests(self):
        drains = []
        allocator = self._allocator(
            [64],
            SimpleNamespace(run_deferred_releases=lambda block=False: drains.append(1)),
        )
        self.assertTrue(allocator.check_decode_capacity(num_tokens=64, tree_cache=None))
        self.assertEqual(drains, [])

    def test_a_shortfall_that_remains_is_reported(self):
        allocator = self._allocator(
            [0], SimpleNamespace(run_deferred_releases=lambda block=False: None)
        )
        self.assertFalse(
            allocator.check_decode_capacity(num_tokens=64, tree_cache=None)
        )


if __name__ == "__main__":
    unittest.main()
