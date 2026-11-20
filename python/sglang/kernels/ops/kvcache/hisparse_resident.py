"""Slot tables for HiSparse requests whose KV stays in the device pool (resident requests)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mask_resident_seq_lens(
    seq_lens,
    req_pool_indices,
    resident,
    output,
    count,
    NUM_STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    query = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = query < count
    rid = tl.load(req_pool_indices + query // NUM_STEPS, valid, other=0).to(tl.int64)
    is_resident = tl.load(resident + rid, valid, other=0) != 0
    length = tl.load(seq_lens + query, valid, other=0)
    tl.store(output + query, tl.where(is_resident, 0, length), valid)


def mask_resident_seq_lens(
    *,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    resident: torch.Tensor,
) -> torch.Tensor:
    """Return `seq_lens` with the queries of resident requests set to length zero.

    The HiSparse swap-in kernels select nothing for a zero-length query and leave its
    request's hot-buffer state alone, which is what a request without a hot buffer needs.

    Tensor layout:
      seq_lens: int32/int64 [num_reqs * num_steps], contiguous. num_steps is 1 in plain
        decode and the number of verified tokens under speculative decoding.
      req_pool_indices: int64 [num_reqs], contiguous. Request slot of each batch row.
      resident: int8 [num_request_slots], contiguous. Nonzero marks a resident request.
    """
    num_reqs = req_pool_indices.shape[0]
    output = torch.empty_like(seq_lens)
    count = seq_lens.numel()
    if count == 0:
        return output
    assert seq_lens.ndim == 1 and seq_lens.is_contiguous()
    assert count % num_reqs == 0
    assert req_pool_indices.dtype == torch.int64 and req_pool_indices.is_contiguous()
    assert resident.dtype == torch.int8 and resident.is_contiguous()
    _mask_resident_seq_lens[(triton.cdiv(count, 128),)](
        seq_lens,
        req_pool_indices,
        resident,
        output,
        count,
        NUM_STEPS=count // num_reqs,
        BLOCK=128,
    )
    return output


@triton.jit
def _translate_resident_rows(
    top_k,
    top_k_stride,
    req_pool_indices,
    seq_lens,
    resident,
    req_to_token,
    req_to_token_stride,
    mapping,
    num_real_reqs,
    output,
    output_stride,
    num_top_k,
    NUM_STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    query = tl.program_id(0)
    req = query // NUM_STEPS
    if req >= tl.load(num_real_reqs):
        return
    rid = tl.load(req_pool_indices + req).to(tl.int64)
    if tl.load(resident + rid) == 0:
        return
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    in_row = column < num_top_k
    position = tl.load(top_k + query * top_k_stride + column, in_row, other=-1).to(
        tl.int64
    )
    selected = in_row & (position >= 0) & (position < tl.load(seq_lens + query))
    logical = tl.load(
        req_to_token + rid * req_to_token_stride + position, selected, other=0
    ).to(tl.int64)
    slot = tl.load(mapping + logical, selected, other=-1)
    tl.store(
        output + query * output_stride + column,
        tl.where(selected, slot, -1).to(tl.int32),
        in_row,
    )


def translate_resident_rows(
    *,
    top_k: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    resident: torch.Tensor,
    req_to_token: torch.Tensor,
    mapping: torch.Tensor,
    num_real_reqs: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Write the rows attention reads for resident requests' selected positions.

    Tensor layout:
      top_k: int32 [num_reqs, num_top_k] or [num_reqs, num_steps, num_top_k]; each query's
        positions are contiguous. Negative entries are padding.
      req_pool_indices: int64 [num_reqs], contiguous. Request slot of each batch row.
      seq_lens: int32/int64 [num_reqs * num_steps], contiguous. A position at or past its
        query's length is not selected.
      resident: int8 [num_request_slots], contiguous. Nonzero marks a resident request.
      req_to_token: int32 [num_request_slots, max_context_len], rows contiguous. Logical
        slot of each position.
      mapping: int64 [num_logical_slots], contiguous. mapping[logical_slot] = the row of
        the layer's KV buffer that holds the slot.
      num_real_reqs: int32 [1]. Batch rows at or past it are CUDA-graph padding.
      output: int32, the shape of top_k. Queries of resident requests get
        mapping[req_to_token[request, position]], and -1 where nothing is selected; every
        other query is left as it is.
    """
    num_reqs = req_pool_indices.shape[0]
    if num_reqs == 0:
        return
    num_steps = 1 if top_k.ndim == 2 else top_k.shape[1]
    num_top_k = top_k.shape[-1]
    assert top_k.dtype == torch.int32 and output.dtype == torch.int32
    assert top_k.shape == output.shape and top_k.shape[0] == num_reqs
    assert top_k.stride(-1) == 1 and output.stride(-1) == 1
    if top_k.ndim == 3:
        # Queries are addressed as one flat [num_reqs * num_steps] axis.
        assert top_k.stride(0) == num_steps * top_k.stride(1)
        assert output.stride(0) == num_steps * output.stride(1)
    assert req_pool_indices.dtype == torch.int64 and req_pool_indices.is_contiguous()
    assert seq_lens.numel() == num_reqs * num_steps and seq_lens.is_contiguous()
    assert resident.dtype == torch.int8 and resident.is_contiguous()
    assert req_to_token.dtype == torch.int32 and req_to_token.stride(1) == 1
    assert mapping.dtype == torch.int64 and mapping.is_contiguous()
    assert num_real_reqs.dtype == torch.int32
    block = 256
    _translate_resident_rows[(num_reqs * num_steps, triton.cdiv(num_top_k, block))](
        top_k,
        top_k.stride(-2),
        req_pool_indices,
        seq_lens,
        resident,
        req_to_token,
        req_to_token.stride(0),
        mapping,
        num_real_reqs,
        output,
        output.stride(-2),
        num_top_k,
        NUM_STEPS=num_steps,
        BLOCK=block,
    )


@triton.jit
def _resident_page_table(
    page_table,
    page_table_stride,
    mapping,
    seq_lens,
    output,
    output_stride,
    num_pages,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    page = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(seq_lens + row).to(tl.int64)
    used = (page < (length + PAGE_SIZE - 1) // PAGE_SIZE) & (page < num_pages)
    logical_page = tl.load(page_table + row * page_table_stride + page, used, other=0)
    logical_page = tl.maximum(logical_page.to(tl.int64), 0)
    slot = tl.load(mapping + logical_page * PAGE_SIZE, used, other=0)
    tl.store(
        output + row * output_stride + page, (slot // PAGE_SIZE).to(tl.int32), used
    )


def resident_page_table(
    *,
    page_table: torch.Tensor,
    mapping: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Map a batch's logical page table to the pages attention reads.

    A resident request's logical pages are mirrored page for page, so the row of position
    p is output[row, p // page_size] * page_size + p % page_size.

    Tensor layout:
      page_table: int32 [>= num_rows, num_pages], rows contiguous. Logical page of each
        page of each batch row; negative entries are padding.
      mapping: int64 [num_logical_slots], contiguous. mapping[logical_slot] = the row of
        the layer's KV buffer that holds the slot.
      seq_lens: int32/int64 [num_rows].
      output: int32 [num_rows, num_pages]. Entries past a row's last page are left
        uninitialized: the top-k kernel reads output[row, p // page_size] for p < length.
    """
    num_rows = seq_lens.shape[0]
    num_pages = page_table.shape[1]
    output = torch.empty(
        (num_rows, num_pages), dtype=torch.int32, device=page_table.device
    )
    if num_rows == 0 or num_pages == 0:
        return output
    assert page_table.dtype == torch.int32 and page_table.stride(1) == 1
    assert page_table.shape[0] >= num_rows
    assert mapping.dtype == torch.int64 and mapping.is_contiguous()
    block = 1024
    _resident_page_table[(num_rows, triton.cdiv(num_pages, block))](
        page_table,
        page_table.stride(0),
        mapping,
        seq_lens,
        output,
        output.stride(0),
        num_pages,
        PAGE_SIZE=page_size,
        BLOCK=block,
    )
    return output
