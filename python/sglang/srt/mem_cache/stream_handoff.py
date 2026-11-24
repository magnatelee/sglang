"""Tensors made on one device stream and read on another.

Two things can go wrong, and neither shows on the host. Lifetime: PyTorch returns a dropped tensor's memory to
the stream that allocated it, which may hand it out again at once; a reader queued on another stream that has
not run yet then reads whatever was written there since. Order: a reader on a stream that does not follow the
maker's stream can run before the maker's kernels have, and reads what the memory held before. The scheduler's
loop happens to order most pairs (a launch waits for the schedule stream, the schedule stream waits for the
launched batch, a release is followed by its caller), but only for the loop as it is.

Every tensor that one stream leaves for another is marked by its maker with `staged` and taken by its reader
through `handed_over`, which waits for the maker on the device (never on the host) and holds the memory for
the reader: staged free pages and the free list itself, kept scratch blocks, held buffer slots, a seal's
picks, a group move's plan.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.utils import get_device_module

device_module = get_device_module()

_MADE = "_handoff_made"  # the event behind the kernels that compute the tensor, on the maker's stream
_HELD = "_handoff_held"  # the streams that already hold the tensor's memory


def staged(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """The current stream has queued what computes `tensor` (from inputs it took through `handed_over`),
    and another stream may read it: remember where on this stream it is complete."""
    if tensor is not None and tensor.is_cuda:
        made = device_module.Event()
        made.record(device_module.current_stream())
        tensor.__dict__[_MADE] = made
    return tensor


def handed_over(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """The current stream reads `tensor`: it runs after the maker's kernels (a wait on the device; the host
    does not wait), and the tensor's memory is not handed out again before this stream has read it."""
    if tensor is None or not tensor.is_cuda:
        return tensor
    stream = device_module.current_stream()
    marks = tensor.__dict__
    made = marks.get(_MADE)
    if made is not None:
        if made.query():
            del marks[_MADE]  # complete for every reader from now on
        else:
            stream.wait_event(made)
    held = marks.get(_HELD)
    if held is None:
        held = marks[_HELD] = set()
    if stream not in held:
        tensor.record_stream(stream)
        held.add(stream)
    return tensor
