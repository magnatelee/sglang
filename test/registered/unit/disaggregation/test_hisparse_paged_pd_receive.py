"""SGLANG_HISPARSE_PAGED_PD_RECEIVE: a paged HiSparse decode receives a transferred prompt in
the device frames of its pool and admits the request as resident. Off (default): main's
direct-to-host receive, unchanged."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from sglang.srt.disaggregation import decode as decode_mod
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    alloc_for_decode_prealloc_hisparse,
)
from sglang.srt.disaggregation.common.utils import build_dcp_token_transfer_plan
from sglang.srt.disaggregation.mooncake.conn import dcp_index_k_dst_rows
from sglang.srt.disaggregation.utils import TransferBackend
from sglang.srt.environ import envs
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.mem_cache.hisparse_shard import HiSparseShardView
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(req_pool_idx: int, extend_end: int):
    return SimpleNamespace(
        rid=f"r{req_pool_idx}",
        kv=SimpleNamespace(req_pool_idx=req_pool_idx, kv_allocated_len=extend_end),
        extend_end=extend_end,
        hisparse_staging=True,
    )


def _coordinator(count: int = 1):
    """A coordinator with the state the two admissions of a transferred prompt touch."""
    c = object.__new__(HiSparseCoordinator)
    c.shard = HiSparseShardView(count=count, rank=0)
    c.enable_resident = True
    c.hs_exclusive = True
    c.hs_spec_res = True
    c.hs_resident_only = True
    c.page_size = 64
    c._resident = {}
    c.req_resident = torch.zeros(8, dtype=torch.int32)
    c.req_device_buffer_size = torch.full((8,), 7, dtype=torch.int32)
    c._hs_staged_len = [0] * 8
    c._hs_dev_len = [0] * 8
    c._skip_first_backup = [False] * 8
    c.req_to_host_pool_allocated_len = torch.zeros(8, dtype=torch.int64)
    c.paging = SimpleNamespace(
        xstats=[0] * 8,
        host_len=[0] * 8,
        registered_len=[9] * 8,
        registered=[],
    )
    c.paging.register = lambda rid, req: c.paging.registered.append(rid)
    return c


def _state(c, rid):
    return (
        sorted(c._resident),
        int(c.req_resident[rid]),
        int(c.req_device_buffer_size[rid]),
        c._hs_staged_len[rid],
        c._hs_dev_len[rid],
        c._skip_first_backup[rid],
        c.paging.host_len[rid],
        c.paging.registered_len[rid],
        list(c.paging.registered),
        c.paging.xstats[5],
    )


def _local_path(c, req):
    """What a local prefill leaves when nothing is copied to host: the staged length that
    admit_request_into_staging records, then move_ready_reqs' resident branch."""
    rid = req.kv.req_pool_idx
    c._hs_staged_len[rid] = c.shard.extent(req.extend_end)
    c.paging.xstats[5] += 1  # _exclusive_staged_len: nothing copied
    c._make_resident(req)
    c._skip_first_backup[rid] = True
    req.hisparse_staging = False


class TestAdmitRequestReceived(CustomTestCase):
    def test_same_state_as_a_local_prefill_that_copies_nothing(self):
        for count, length in ((1, 5000), (4, 5000), (4, 5003), (4, 63)):
            a, b = _coordinator(count), _coordinator(count)
            ra, rb = _req(3, length), _req(3, length)
            _local_path(a, ra)
            b.admit_request_received(rb)
            self.assertEqual(_state(a, 3), _state(b, 3), (count, length))
            self.assertFalse(rb.hisparse_staging)
            self.assertIn(3, b._resident)

    def test_split_stages_rows_not_tokens(self):
        c = _coordinator(4)
        c.admit_request_received(_req(2, 5003))
        self.assertEqual(c._hs_staged_len[2], 1251)  # ceil(5003 / 4) rows on every shard
        self.assertEqual(c._hs_dev_len[2], 1251)

    def test_mutant_without_the_staged_length_differs(self):
        a, b = _coordinator(4), _coordinator(4)
        _local_path(a, _req(1, 5000))
        req = _req(1, 5000)
        b._make_resident(req)  # the mutant: resident without its staged prefix
        self.assertNotEqual(_state(a, 1), _state(b, 1))


class TestPreallocForTransfer(CustomTestCase):
    def _allocator(self, out):
        alloc = MagicMock()
        alloc.device = "cpu"
        alloc.alloc_extend.return_value = out
        alloc.alloc_logical_only.return_value = torch.arange(3)
        return alloc

    def _kv(self):
        return SimpleNamespace(allocate_prealloc=MagicMock())

    def test_resident_takes_frames_through_alloc_extend(self):
        out = torch.arange(700)
        alloc = self._allocator(out)
        req = SimpleNamespace(kv=self._kv())
        got = alloc_for_decode_prealloc_hisparse(
            alloc, req=req, fill_len=700, uses_swa_tail=False, swa_tail_len=700, resident=True
        )
        self.assertIs(got, out)
        alloc.alloc_logical_only.assert_not_called()
        kwargs = alloc.alloc_extend.call_args.kwargs
        self.assertEqual(kwargs["extend_num_tokens"], 700)
        self.assertEqual(kwargs["prefix_lens_cpu"].tolist(), [0])
        self.assertEqual(kwargs["seq_lens_cpu"].tolist(), [700])
        req.kv.allocate_prealloc.assert_called_once_with(700)

    def test_default_is_the_direct_to_host_allocation(self):
        alloc = self._allocator(None)
        req = SimpleNamespace(kv=self._kv())
        alloc_for_decode_prealloc_hisparse(
            alloc, req=req, fill_len=700, uses_swa_tail=False, swa_tail_len=700
        )
        alloc.alloc_extend.assert_not_called()
        alloc.alloc_logical_only.assert_called_once()


class TestIndexKeyRowsOfADcpRelayout(CustomTestCase):
    """The sender of a DCP relayout placed the index keys by the KV destination rows.
    A paged decode publishes device frames for KV and keeps its index keys at the
    logical rows: with one vector the keys landed at the frame rows (wrong answers once
    frames and logical rows differ). The keys must follow their own pages."""

    KW = dict(
        physical_page_size=4,
        dcp_size=2,
        dcp_rank=1,
        src_page_offset=0,
        decode_prefix_len=0,
        num_kv_tokens=14,
        draft_rows_collapsed=False,
    )

    def _plan(self, dst_pages):
        return build_dcp_token_transfer_plan(
            np.array([5, 6, 2, 9], dtype=np.int32), dst_pages, **self.KW
        )

    def test_keys_follow_their_own_pages(self):
        src = np.array([5, 6, 2, 9], dtype=np.int32)
        frames = np.array([40, 41], dtype=np.int32)  # where KV lands
        logical = np.array([7, 3], dtype=np.int32)  # where the indexer reads keys
        plan = self._plan(frames)
        rows = dcp_index_k_dst_rows(plan, src, logical, **self.KW)
        want = self._plan(logical).target_dst_token_indices
        self.assertEqual(rows.tolist(), want.tolist())
        self.assertNotEqual(rows.tolist(), plan.target_dst_token_indices.tolist())
        # Same tokens, same offsets inside a page: only the page differs.
        kv = plan.target_dst_token_indices
        self.assertEqual((rows % 4).tolist(), (kv % 4).tolist())
        self.assertEqual(sorted(set((rows // 4).tolist())), [3, 7])

    def test_without_separate_pages_keys_use_the_kv_rows(self):
        src = np.array([5, 6, 2, 9], dtype=np.int32)
        plan = self._plan(np.array([40, 41], dtype=np.int32))
        rows = dcp_index_k_dst_rows(plan, src, None, **self.KW)
        self.assertIs(rows, plan.target_dst_token_indices)


class TestSwitchAndSupportCheck(CustomTestCase):
    def test_switch_is_off_by_default(self):
        self.assertFalse(envs.SGLANG_HISPARSE_PAGED_PD_RECEIVE.get())
        self.assertFalse(DecodePreallocQueue.hisparse_resident_receive)

    def _pool(self, peer_writable):
        pool = object.__new__(HiSparseDSATokenToKVPool)
        pool.hs_peer_writable = peer_writable
        return pool

    def _queue(self, *, resident_only, peer_writable, backend=TransferBackend.MOONCAKE,
               draft_pool=None, ring=None):
        q = object.__new__(DecodePreallocQueue)
        q.scheduler = SimpleNamespace(
            hisparse_coordinator=SimpleNamespace(
                hs_resident_only=resident_only,
                mem_pool_device=self._pool(peer_writable),
            )
        )
        q.transfer_backend = backend
        q.draft_token_to_kv_pool = draft_pool
        q.draft_kv_ring = ring
        return q

    def test_supported_configurations(self):
        with unittest.mock.patch.object(
            decode_mod, "get_parallel", return_value=SimpleNamespace(pp_size=1)
        ):
            self._queue(resident_only=True, peer_writable=True)._check_resident_receive_supported()
            self._queue(
                resident_only=True, peer_writable=True, draft_pool=object(), ring=object()
            )._check_resident_receive_supported()
            self._queue(
                resident_only=True, peer_writable=False, backend=TransferBackend.FAKE
            )._check_resident_receive_supported()

    def test_refusals(self):
        cases = [
            dict(resident_only=False, peer_writable=True),  # buffered requests exist
            dict(resident_only=True, peer_writable=False),  # pool not exportable
            dict(resident_only=True, peer_writable=True, draft_pool=object()),  # draft rows by token id
            dict(resident_only=False, peer_writable=False, backend=TransferBackend.FAKE),
        ]
        with unittest.mock.patch.object(
            decode_mod, "get_parallel", return_value=SimpleNamespace(pp_size=1)
        ):
            for kw in cases:
                with self.assertRaises(ValueError, msg=str(kw)):
                    self._queue(**kw)._check_resident_receive_supported()
        with unittest.mock.patch.object(
            decode_mod, "get_parallel", return_value=SimpleNamespace(pp_size=2)
        ):
            with self.assertRaises(ValueError):
                self._queue(resident_only=True, peer_writable=True)._check_resident_receive_supported()


if __name__ == "__main__":
    unittest.main()
