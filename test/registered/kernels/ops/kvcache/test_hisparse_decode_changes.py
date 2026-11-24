"""Two decode-path switches of HiSparse paged residency against the paths they replace: the write slots of a
forward translated once, and the row cache's plan in fewer kernels."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-small")

PAGE = 64
BLOCK = 1024
DEV_PAGES = 96
HOST_PAGES = 512
HOST_F0 = DEV_PAGES + 8  # first host page, in page units of the row index space
HOST0 = HOST_F0 * PAGE
HOST_END = HOST0 + HOST_PAGES * PAGE
CACHE_ROWS = 4096
LOGC = 12


def _write_pool(*, once: bool):
    from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool

    pool = HiSparseDSATokenToKVPool.__new__(HiSparseDSATokenToKVPool)
    pool.full_to_hisparse_device_index_mapping = (
        torch.randperm(4096, generator=torch.Generator().manual_seed(1)).cuda() + 7
    )
    pool._hs_wloc_once = once
    pool._hs_wloc_memo = None
    pool._hs_wloc_stats = [0, 0]
    translations = []

    def translate(loc):
        translations.append(loc)
        return pool.full_to_hisparse_device_index_mapping[loc.to(torch.int64)]

    pool.translate_loc_to_hisparse_device = translate
    return pool, translations


class TestHiSparseWriteSlotsOnce(CustomTestCase):
    def _expect(self, pool, loc):
        return pool.full_to_hisparse_device_index_mapping[loc.to(torch.int64)]

    def test_one_translation_serves_every_layer_of_a_forward(self):
        pool, translations = _write_pool(once=True)
        loc = torch.arange(100, 164, device="cuda")
        slots = [pool.hs_write_slots(loc, layer_id) for layer_id in range(4)]
        self.assertEqual(len(translations), 1)
        for table in slots:
            self.assertIs(table, slots[0])
        self.assertTrue(torch.equal(slots[0], self._expect(pool, loc)))

    def test_the_next_forward_translates_again(self):
        """Layer ids start over: the map may have changed through a kernel, which no version counter sees."""
        pool, translations = _write_pool(once=True)
        loc = torch.arange(100, 164, device="cuda")
        for layer_id in range(3):
            pool.hs_write_slots(loc, layer_id)
        mapping = pool.full_to_hisparse_device_index_mapping
        version = mapping._version
        mapping.data[100:164] = torch.arange(9000, 9064, device="cuda")  # as a kernel writes it
        self.assertEqual(mapping._version, version)
        again = pool.hs_write_slots(loc, 0)
        self.assertEqual(len(translations), 2)
        self.assertTrue(torch.equal(again, torch.arange(9000, 9064, device="cuda")))

    def test_a_changed_table_is_not_served_from_the_last_translation(self):
        pool, translations = _write_pool(once=True)
        loc = torch.arange(100, 164, device="cuda")
        pool.hs_write_slots(loc, 0)
        loc += 1  # the same tensor, other locations
        moved = pool.hs_write_slots(loc, 1)
        self.assertEqual(len(translations), 2)
        self.assertTrue(torch.equal(moved, self._expect(pool, loc)))
        other = loc.clone()  # equal contents, another tensor
        pool.hs_write_slots(other, 2)
        self.assertEqual(len(translations), 3)
        pool.full_to_hisparse_device_index_mapping[5] = 1  # the map written inside the forward
        fresh = pool.hs_write_slots(other, 3)
        self.assertEqual(len(translations), 4)
        self.assertTrue(torch.equal(fresh, self._expect(pool, other)))

    def test_switch_off_translates_at_every_layer(self):
        pool, translations = _write_pool(once=False)
        loc = torch.arange(100, 164, device="cuda")
        for layer_id in range(4):
            pool.hs_write_slots(loc, layer_id)
        self.assertEqual(len(translations), 4)
        self.assertEqual(pool._hs_wloc_stats, [0, 0])


def _cache_state(seed):
    """One layer group's redirects and row cache with entries already installed, and a slot table of one
    program (the claim kernel's race between programs is not what is compared here)."""
    gen = torch.Generator().manual_seed(seed)
    i32 = dict(dtype=torch.int32, device="cuda")
    total = DEV_PAGES + HOST_PAGES
    redir = torch.full((total + 1,), -1, **i32)
    holes = torch.randperm(DEV_PAGES, generator=gen)[:24]  # frames whose home page the group reads from host
    hole_hosts = torch.randperm(HOST_PAGES, generator=gen)[:24]
    redir[holes.cuda()] = (DEV_PAGES + hole_hosts).to(torch.int32).cuda()
    guests = torch.randperm(HOST_PAGES, generator=gen)[24:40]  # host pages the group reads from a frame
    redir[(DEV_PAGES + guests).cuda()] = holes[:16].to(torch.int32).cuda()
    n = 900
    frames = torch.randint(1, DEV_PAGES, (n,), generator=gen)
    hosts = HOST_F0 + torch.randint(0, HOST_PAGES, (n,), generator=gen)
    on_host = torch.rand(n, generator=gen) < 0.6
    slots = torch.where(on_host, hosts, frames) * PAGE + torch.randint(0, PAGE, (n,), generator=gen)
    slots[torch.rand(n, generator=gen) < 0.05] = -1  # padding
    state = dict(
        redir=redir,
        cnt=torch.zeros(total + 1, **i32),
        tag=torch.full((CACHE_ROWS,), -1, **i32),
        tep=torch.zeros(CACHE_ROWS, **i32),
        ep=torch.ones(HOST_PAGES + 1, **i32),
        used=torch.zeros(CACHE_ROWS, **i32),
        cval=torch.zeros(CACHE_ROWS, **i32),
        cst=torch.zeros(CACHE_ROWS, **i32),
        stamp=torch.zeros(1, **i32),
        scr=(100000 + torch.randperm(CACHE_ROWS, generator=gen)).to(torch.int32).cuda(),
        src=torch.zeros(CACHE_ROWS, **i32),
        dst=torch.zeros(CACHE_ROWS, **i32),
        tot=torch.zeros(1, **i32),
        stat=torch.zeros(3, dtype=torch.int64, device="cuda"),
    )
    return state, slots.to(torch.int32).cuda().contiguous()


def _plan(state, slots, *, fused: bool, cap: int, claims=None):
    """One redirect and one plan of the row cache over `slots`, as the paging manager launches them.
    claims: (cval, cst) another plan over the same state left; taken instead of running the claim kernel,
    whose winner among rows of one cache line is a race between threads."""
    from sglang.kernels.ops.kvcache import hisparse_paging as K

    s = state
    n = slots.numel()
    nb = (n + BLOCK - 1) // BLOCK
    cb = CACHE_ROWS // BLOCK
    out = torch.empty_like(slots)
    s["stamp"].add_(1)
    red = (slots, out, n, s["redir"], s["cnt"], HOST_F0, DEV_PAGES, DEV_PAGES + HOST_PAGES)
    row = (s["tag"], s["tep"], s["ep"], s["used"])
    if fused:
        K.redirect_slots_mark[(nb,)](
            *red, *row, s["stamp"], HOST0, HOST_END, LOGC=LOGC, PAGE=PAGE, BLOCK=BLOCK
        )
    else:
        K.redirect_slots[(nb,)](*red, PAGE=PAGE, BLOCK=BLOCK)
        K.row_cache_mark[(nb,)](
            out, n, *row, s["stamp"], HOST0, HOST_END, LOGC=LOGC, PAGE=PAGE, BLOCK=BLOCK
        )
    if claims is None:
        K.row_cache_claim[(nb,)](
            out, n, *row, s["cval"], s["cst"], s["stamp"], HOST0, HOST_END,
            LOGC=LOGC, PAGE=PAGE, BLOCK=BLOCK,
        )  # fmt: skip
    else:
        s["cval"].copy_(claims[0])
        s["cst"].copy_(claims[1])
    cnt = torch.empty(cb, dtype=torch.int32, device="cuda")
    K.row_cache_count[(cb,)](s["cst"], s["stamp"], cnt, BLOCK=BLOCK)
    lists = (s["scr"], s["ep"], s["tag"], s["tep"], s["src"], s["dst"], HOST0, cap)
    if fused:
        K.row_cache_compact_scan[(cb,)](
            s["cst"], s["cval"], s["stamp"], cnt, s["tot"], *lists,
            PAGE=PAGE, BLOCK=BLOCK, NBLK=cb,
        )  # fmt: skip
    else:
        c64 = cnt.to(torch.int64)
        off = (torch.cumsum(c64, 0) - c64).contiguous()
        s["tot"].copy_(c64.sum().clamp(max=cap).to(torch.int32).reshape(1))
        K.row_cache_compact[(cb,)](
            s["cst"], s["cval"], s["stamp"], off, *lists, PAGE=PAGE, BLOCK=BLOCK
        )
    out_a, out_r = torch.empty_like(out), torch.empty_like(out)
    K.row_cache_resolve[(nb,)](
        out, out_a, out_r, n, *row, s["cval"], s["cst"], s["stamp"], s["scr"], s["stat"],
        HOST0, HOST_END, LOGC=LOGC, PAGE=PAGE, BLOCK=BLOCK,
    )  # fmt: skip
    return out, out_a, out_r


class TestHiSparseRowCacheFusedPlan(CustomTestCase):
    def _compare(self, seed, cap):
        """Three plans with the fused kernels; beside each, the unfused kernels on a copy of the state
        the plan started from, given the claims the fused plan left (a claim the compacting kernel
        voided for lack of room in the list is voided in both: the voided ones are the list's tail)."""
        fused, slots = _cache_state(seed)
        for plan in range(3):  # installs, then hits on them, then both
            if plan == 2:
                slots = torch.roll(slots, 300) + 1
                slots[slots == 0] = -1
            unfused = {key: value.clone() for key, value in fused.items()}
            got = _plan(fused, slots, fused=True, cap=cap)
            want = _plan(
                unfused, slots, fused=False, cap=cap, claims=(fused["cval"], fused["cst"])
            )
            for name, a, b in zip(("redirected", "indexer", "reusing"), want, got):
                self.assertTrue(torch.equal(a, b), f"plan {plan}: {name} table")
            total = int(unfused["tot"])
            self.assertEqual(total, int(fused["tot"]), f"plan {plan}: install list length")
            for key in ("cnt", "tag", "tep", "used", "cval", "cst", "stat", "stamp"):
                self.assertTrue(
                    torch.equal(unfused[key], fused[key]), f"plan {plan}: {key}"
                )
            for key in ("src", "dst"):
                self.assertTrue(
                    torch.equal(unfused[key][:total], fused[key][:total]),
                    f"plan {plan}: install list {key}",
                )
        return fused

    def test_plans_equal_the_unfused_kernels(self):
        for seed in range(4):
            with self.subTest(seed=seed):
                state = self._compare(seed, cap=CACHE_ROWS)
                hits, installs, in_place = state["stat"].tolist()
                self.assertGreater(hits, 0)
                self.assertGreater(installs, 0)

    def test_claims_past_the_list_capacity_are_void_in_both(self):
        state = self._compare(0, cap=64)
        self.assertEqual(int(state["tot"]), 64)
        self.assertGreater(int(state["stat"][2]), 0)  # rows read in place

    def test_a_wrong_block_offset_is_seen(self):
        """The comparison notices an install list built from other offsets than the block counts give."""
        from sglang.kernels.ops.kvcache import hisparse_paging as K

        fused, slots = _cache_state(0)
        unfused = {key: value.clone() for key, value in fused.items()}
        _plan(fused, slots, fused=True, cap=CACHE_ROWS)
        _plan(unfused, slots, fused=False, cap=CACHE_ROWS, claims=(fused["cval"], fused["cst"]))
        total = int(fused["tot"])
        self.assertTrue(torch.equal(unfused["src"][:total], fused["src"][:total]))
        # the same compaction with every block's count taken as zero: all blocks write from offset 0
        again = {key: value.clone() for key, value in unfused.items()}
        cb = CACHE_ROWS // BLOCK
        zero = torch.zeros(cb, dtype=torch.int32, device="cuda")
        K.row_cache_compact_scan[(cb,)](
            again["cst"], again["cval"], again["stamp"], zero, again["tot"], again["scr"],
            again["ep"], again["tag"], again["tep"], again["src"], again["dst"], HOST0, CACHE_ROWS,
            PAGE=PAGE, BLOCK=BLOCK, NBLK=cb,
        )  # fmt: skip
        self.assertNotEqual(int(again["tot"]), total)


if __name__ == "__main__":
    unittest.main()
