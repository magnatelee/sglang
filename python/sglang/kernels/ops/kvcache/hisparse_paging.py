"""Triton kernels of HiSparse paged residency: per-group slot redirects, group page copies, the host-row cache.

Frame indices: device pages first (slot // PAGE), then host pages (DEV + slot // PAGE - HOST_F0). A host row x (its
index in the host pool) lives in row-cache slot ((x * 2654435761) mod 2^32) >> (32 - LOGC).
"""

import triton
import triton.language as tl


# out[i] = the slot a layer group reads for slot in[i]: the same slot unless the group redirected its
# frame (redir[frame] >= 0), then the same row offset in that frame. Counts the frame that is read.
@triton.jit
def redirect_slots(
    in_ptr,
    out_ptr,
    n,
    redir_ptr,
    cnt_ptr,
    host_f0,
    dev_pages,
    total,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    s = tl.load(in_ptr + offs, mask=m, other=-1).to(tl.int64)
    f = s // PAGE
    off = s - f * PAGE
    u = tl.where(f >= host_f0, f - host_f0 + dev_pages, f)
    ok = m & (s >= 0) & (u >= 0) & (u < total)
    r = tl.load(redir_ptr + u, mask=ok, other=-1).to(tl.int64)
    u2 = tl.where(r >= 0, r, u)
    f2 = tl.where(u2 >= dev_pages, u2 - dev_pages + host_f0, u2)
    s2 = tl.where(ok, f2 * PAGE + off, s)
    tl.store(out_ptr + offs, s2.to(tl.int32), mask=m)
    tl.atomic_add(cnt_ptr + u2, tl.full([BLOCK], 1, tl.int32), mask=ok, sem="relaxed")


# redirect_slots and row_cache_mark in one pass over the slots (SGLANG_HISPARSE_PAGED_ROWCACHE_FUSED): the same
# output, the same counters and the same used stamps as the two kernels one after the other.
@triton.jit
def redirect_slots_mark(
    in_ptr,
    out_ptr,
    n,
    redir_ptr,
    cnt_ptr,
    host_f0,
    dev_pages,
    total,
    tag_ptr,
    tep_ptr,
    ep_ptr,
    used_ptr,
    stamp_ptr,
    host0,
    host_end,
    LOGC: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    s = tl.load(in_ptr + offs, mask=m, other=-1).to(tl.int64)
    f = s // PAGE
    off = s - f * PAGE
    u = tl.where(f >= host_f0, f - host_f0 + dev_pages, f)
    ok = m & (s >= 0) & (u >= 0) & (u < total)
    r = tl.load(redir_ptr + u, mask=ok, other=-1).to(tl.int64)
    u2 = tl.where(r >= 0, r, u)
    f2 = tl.where(u2 >= dev_pages, u2 - dev_pages + host_f0, u2)
    s2 = tl.where(ok, f2 * PAGE + off, s)
    s32 = s2.to(tl.int32)
    tl.store(out_ptr + offs, s32, mask=m)
    tl.atomic_add(cnt_ptr + u2, tl.full([BLOCK], 1, tl.int32), mask=ok, sem="relaxed")
    # the mark pass on what was just stored (row_cache_mark reads it back as int32)
    t = s32.to(tl.int64)
    h = m & (t >= host0) & (t < host_end)
    x = tl.where(h, t - host0, 0)
    i = ((x * 2654435761) & 0xFFFFFFFF) >> (32 - LOGC)
    hit = (
        h
        & (tl.load(tag_ptr + i, mask=h, other=-1).to(tl.int64) == x)
        & (
            tl.load(tep_ptr + i, mask=h, other=-1)
            == tl.load(ep_ptr + x // PAGE, mask=h, other=-2)
        )
    )
    stamp = tl.load(stamp_ptr)
    tl.store(used_ptr + i, tl.zeros([BLOCK], tl.int32) + stamp, mask=hit)


@triton.jit
def row_cache_mark(
    in_ptr,
    n,
    tag_ptr,
    tep_ptr,
    ep_ptr,
    used_ptr,
    stamp_ptr,
    host0,
    host_end,
    LOGC: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    s = tl.load(in_ptr + offs, mask=m, other=-1).to(tl.int64)
    h = m & (s >= host0) & (s < host_end)
    x = tl.where(h, s - host0, 0)
    i = ((x * 2654435761) & 0xFFFFFFFF) >> (32 - LOGC)
    hit = (
        h
        & (tl.load(tag_ptr + i, mask=h, other=-1).to(tl.int64) == x)
        & (
            tl.load(tep_ptr + i, mask=h, other=-1)
            == tl.load(ep_ptr + x // PAGE, mask=h, other=-2)
        )
    )
    stamp = tl.load(stamp_ptr)
    tl.store(used_ptr + i, tl.zeros([BLOCK], tl.int32) + stamp, mask=hit)


@triton.jit
def row_cache_claim(
    in_ptr,
    n,
    tag_ptr,
    tep_ptr,
    ep_ptr,
    used_ptr,
    cval_ptr,
    cst_ptr,
    stamp_ptr,
    host0,
    host_end,
    LOGC: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    s = tl.load(in_ptr + offs, mask=m, other=-1).to(tl.int64)
    h = m & (s >= host0) & (s < host_end)
    x = tl.where(h, s - host0, 0)
    i = ((x * 2654435761) & 0xFFFFFFFF) >> (32 - LOGC)
    hit = (
        h
        & (tl.load(tag_ptr + i, mask=h, other=-1).to(tl.int64) == x)
        & (
            tl.load(tep_ptr + i, mask=h, other=-1)
            == tl.load(ep_ptr + x // PAGE, mask=h, other=-2)
        )
    )
    stamp = tl.load(stamp_ptr)
    can = h & (~hit) & (tl.load(used_ptr + i, mask=h, other=0) != stamp)
    tl.store(cval_ptr + i, x.to(tl.int32), mask=can)
    tl.store(cst_ptr + i, tl.zeros([BLOCK], tl.int32) + stamp, mask=can)


@triton.jit
def row_cache_count(cst_ptr, stamp_ptr, cnt_ptr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    j = b * BLOCK + tl.arange(0, BLOCK)
    c = tl.load(cst_ptr + j) == tl.load(stamp_ptr)
    tl.store(cnt_ptr + b, tl.sum(c.to(tl.int32), axis=0))


@triton.jit
def row_cache_compact(
    cst_ptr,
    cval_ptr,
    stamp_ptr,
    off_ptr,
    scr_ptr,
    ep_ptr,
    tag_ptr,
    tep_ptr,
    src_ptr,
    dst_ptr,
    host0,
    cap,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    j = b * BLOCK + tl.arange(0, BLOCK)
    c = tl.load(cst_ptr + j) == tl.load(stamp_ptr)
    k = tl.load(off_ptr + b).to(tl.int64) + tl.cumsum(c.to(tl.int64), axis=0) - 1
    ok = c & (k < cap)
    x = tl.load(cval_ptr + j, mask=c, other=0)
    tl.store(src_ptr + k, (x.to(tl.int64) + host0).to(tl.int32), mask=ok)
    tl.store(dst_ptr + k, tl.load(scr_ptr + j, mask=ok, other=0), mask=ok)
    tl.store(tag_ptr + j, x, mask=ok)
    tl.store(
        tep_ptr + j, tl.load(ep_ptr + x.to(tl.int64) // PAGE, mask=ok, other=0), mask=ok
    )
    tl.store(
        cst_ptr + j, tl.zeros([BLOCK], tl.int32), mask=c & (~ok)
    )  # no room in the list: the claim is void


# row_cache_compact with the block offsets and the list's length computed here from the block counts
# (SGLANG_HISPARSE_PAGED_ROWCACHE_FUSED): off[b] = the counts of the blocks before b, total = min(all, cap).
@triton.jit
def row_cache_compact_scan(
    cst_ptr,
    cval_ptr,
    stamp_ptr,
    cnt_ptr,
    tot_ptr,
    scr_ptr,
    ep_ptr,
    tag_ptr,
    tep_ptr,
    src_ptr,
    dst_ptr,
    host0,
    cap,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
    NBLK: tl.constexpr,
):
    b = tl.program_id(0)
    bb = tl.arange(0, NBLK)
    cn = tl.load(cnt_ptr + bb).to(tl.int64)
    base = tl.sum(tl.where(bb < b, cn, 0), axis=0)
    if b == 0:
        tl.store(tot_ptr, tl.minimum(tl.sum(cn, axis=0), cap).to(tl.int32))
    j = b * BLOCK + tl.arange(0, BLOCK)
    c = tl.load(cst_ptr + j) == tl.load(stamp_ptr)
    k = base + tl.cumsum(c.to(tl.int64), axis=0) - 1
    ok = c & (k < cap)
    x = tl.load(cval_ptr + j, mask=c, other=0)
    tl.store(src_ptr + k, (x.to(tl.int64) + host0).to(tl.int32), mask=ok)
    tl.store(dst_ptr + k, tl.load(scr_ptr + j, mask=ok, other=0), mask=ok)
    tl.store(tag_ptr + j, x, mask=ok)
    tl.store(
        tep_ptr + j, tl.load(ep_ptr + x.to(tl.int64) // PAGE, mask=ok, other=0), mask=ok
    )
    tl.store(
        cst_ptr + j, tl.zeros([BLOCK], tl.int32), mask=c & (~ok)
    )  # no room in the list: the claim is void


@triton.jit
def row_cache_resolve(
    in_ptr,
    outa_ptr,
    outr_ptr,
    n,
    tag_ptr,
    tep_ptr,
    ep_ptr,
    used_ptr,
    cval_ptr,
    cst_ptr,
    stamp_ptr,
    scr_ptr,
    stat_ptr,
    host0,
    host_end,
    LOGC: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    s32 = tl.load(in_ptr + offs, mask=m, other=-1)
    s = s32.to(tl.int64)
    h = m & (s >= host0) & (s < host_end)
    x = tl.where(h, s - host0, 0)
    i = ((x * 2654435761) & 0xFFFFFFFF) >> (32 - LOGC)
    stamp = tl.load(stamp_ptr)
    # a hit is a slot stamped used by the mark pass that still carries this row (compact rewrote only unused slots)
    hit = (
        h
        & (tl.load(used_ptr + i, mask=h, other=0) == stamp)
        & (tl.load(tag_ptr + i, mask=h, other=-1).to(tl.int64) == x)
        & (
            tl.load(tep_ptr + i, mask=h, other=-1)
            == tl.load(ep_ptr + x // PAGE, mask=h, other=-2)
        )
    )
    win = (
        h
        & (~hit)
        & (tl.load(cst_ptr + i, mask=h, other=0) == stamp)
        & (tl.load(cval_ptr + i, mask=h, other=-1).to(tl.int64) == x)
    )
    d = tl.load(scr_ptr + i, mask=h, other=0)
    tl.store(outa_ptr + offs, tl.where(hit, d, s32), mask=m)
    tl.store(outr_ptr + offs, tl.where(hit | win, d, s32), mask=m)
    tl.atomic_add(stat_ptr, tl.sum(hit.to(tl.int64), axis=0))
    tl.atomic_add(stat_ptr + 1, tl.sum(win.to(tl.int64), axis=0))
    tl.atomic_add(stat_ptr + 2, tl.sum((h & (~hit) & (~win)).to(tl.int64), axis=0))


@triton.jit
def row_cache_copy(
    ptrs_ptr,
    gk,
    src_ptr,
    dst_ptr,
    tot_ptr,
    NB: tl.constexpr,
    RPI: tl.constexpr,
    ROWW: tl.constexpr,
    TILE: tl.constexpr,
):
    p = tl.program_id(0)
    total = tl.load(tot_ptr)
    bp = tl.cast(tl.load(ptrs_ptr + gk), tl.pointer_type(tl.uint64))
    r = tl.arange(0, RPI)
    w = tl.arange(0, TILE)
    wm = w < ROWW
    n_it = (total + NB * RPI - 1) // (NB * RPI)
    for it in range(0, n_it):
        k = (it * NB + p) * RPI + r
        km = k < total
        s0 = tl.load(src_ptr + k, mask=km, other=0).to(tl.int64)
        d0 = tl.load(dst_ptr + k, mask=km, other=0).to(tl.int64)
        mm = km[:, None] & wm[None, :]
        v = tl.load(bp + s0[:, None] * ROWW + w[None, :], mask=mm)
        tl.store(bp + d0[:, None] * ROWW + w[None, :], v, mask=mm)


# Pair p: rows [src[p], src[p] + PAGE) -> rows [dst[p], ...) of every layer of group grp[p] (ptrs[grp, k]: the
# layer's base address, 0 = no layer), when ok[p]. Rows of ROW bytes.
@triton.jit
def copy_group_pages(
    ptrs_ptr,
    grp_ptr,
    src_ptr,
    dst_ptr,
    ok_ptr,
    LMAX: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    ROW: tl.constexpr,
    TILE: tl.constexpr,
):
    p = tl.program_id(0)
    k = tl.program_id(1)
    t = tl.program_id(2)
    if tl.load(ok_ptr + p) == 0:
        return
    g = tl.load(grp_ptr + p)
    base = tl.load(ptrs_ptr + g * LMAX + k)
    if base == 0:
        return
    bp = tl.cast(base, tl.pointer_type(tl.uint8))
    s0 = tl.load(src_ptr + p)
    d0 = tl.load(dst_ptr + p)
    off = t * TILE + tl.arange(0, TILE)
    mask = off < PAGE_BYTES
    v = tl.load(bp + s0 * ROW + off, mask=mask)
    tl.store(bp + d0 * ROW + off, v, mask=mask)
