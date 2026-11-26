"""One GPU's share of a context that is split over several GPUs (--dcp-size).

Position p of a request lives on shard p % count as that shard's row p // count, and the
request table holds widened slots: slot s is row s // count of shard s % count. The paged
HiSparse scheme keeps its tables per shard, in rows: a request of L tokens spans
extent(L) rows on every shard, whether or not a shard's own last row is written yet, so
every shard holds the same pages of a request at every step. A row exists from the moment
its anchor slot (the one of shard 0) is allocated until that slot is freed.

count 1 is no split: every method returns its argument, or the same slice.
"""

from __future__ import annotations

import msgspec
import torch

from sglang.srt.environ import envs


class HiSparseShardView(msgspec.Struct, frozen=True):
    count: int = 1  # GPUs a context is split over
    rank: int = 0  # this GPU's shard

    @property
    def split(self) -> bool:
        return self.count > 1

    def extent(self, num_tokens):
        """Rows a request of `num_tokens` spans on every shard (int or tensor)."""
        if self.count == 1:
            return num_tokens
        return (num_tokens + self.count - 1) // self.count

    def whole(self, num_tokens):
        """Rows that are written on every shard (int or tensor)."""
        if self.count == 1:
            return num_tokens
        return num_tokens // self.count

    def tokens(self, num_rows):
        """Tokens that `num_rows` rows of every shard hold together (int or tensor)."""
        return num_rows * self.count

    def row_of(self, slots: torch.Tensor) -> torch.Tensor:
        """The logical row of each widened slot."""
        if self.count == 1:
            return slots
        return slots // self.count

    def anchor_rows(self, slots: torch.Tensor) -> torch.Tensor:
        """The logical rows whose anchor slot is among `slots`, in their order."""
        if self.count == 1:
            return slots
        return slots[slots % self.count == 0] // self.count

    def is_anchor(self, slots: torch.Tensor) -> torch.Tensor:
        """Which of `slots` open a row (bool, one per slot)."""
        if self.count == 1:
            return torch.ones_like(slots, dtype=torch.bool)
        return slots % self.count == 0

    def rows(
        self,
        req_to_token: torch.Tensor,
        req_pool_idx: int,
        start: int,
        end: int,
        step: int = 1,
    ) -> torch.Tensor:
        """Logical rows of a request's row indices [start, end), every `step`-th."""
        if self.count == 1:
            return req_to_token[req_pool_idx, start:end:step]
        n = self.count
        return req_to_token[req_pool_idx, start * n : end * n : step * n] // n


def current_shard_view() -> HiSparseShardView:
    """This worker's share of a split context for the paged HiSparse pools: no split
    unless SGLANG_HISPARSE_PAGED_DCP is on and the worker runs with --dcp-size > 1."""
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if envs.SGLANG_HISPARSE_PAGED_DCP.get() and parallel.attn_dcp_size > 1:
        return HiSparseShardView(
            count=parallel.attn_dcp_size, rank=parallel.attn_dcp_rank
        )
    return HiSparseShardView()
