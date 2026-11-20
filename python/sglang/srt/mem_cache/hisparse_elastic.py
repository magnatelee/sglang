"""HiSparse address ranges for one rank: device KV rows, index keys and host KV rows as CUDA VMM mappings.

The device memory a static configuration reserves for the index keys of tokens that might be admitted (sized for
the logical pool, host_to_device_ratio x the device pool) is lent to the device KV pool until those tokens arrive.

Per rank, in reservations whose base addresses never change (CUDA graphs and kernels keep their pointers):
  KV   layers x (device rows, then the lent rows ("band"), a gap, then the layer's host rows). Device rows
       [0, static pool) stay mapped; the band above is mapped in chunks while lent; host rows are pinned host
       memory mapped for the GPU, so a kernel indexes a host row like a device row.
  IK   indexer layers x the logical pool's key pages, mapped in chunks as logical pages come into use.
The band plus the mapped index keys never exceed the static index-key reservation, so the device memory in use
is the static configuration's.
"""

from __future__ import annotations

import functools
import logging
from typing import Dict, List

import torch
from cuda.bindings import driver as cuda

from sglang.srt.utils.cuda_vmm_utils import (
    align_up,
    check_drv,
    get_allocation_granularity,
    get_device_granularity,
    make_device_allocation_prop,
    make_rw_access_desc,
    tensor_from_pointer,
)

logger = logging.getLogger(__name__)


class VmmChunkPool:
    """Physical chunks of one device, mapped and unmapped at fixed addresses.

    An unmapped chunk is kept and mapped again wherever the next one is needed: memory released to the driver
    could be taken by PyTorch's caching allocator before the next map (cuMemCreate out of memory).
    """

    def __init__(self, device_id: int):
        self.device_id = int(device_id)
        # Not exported to other processes, so no shareable handle type.
        self._prop = make_device_allocation_prop(self.device_id, handle_types=None)
        self._access = [make_rw_access_desc(self.device_id)]
        self._mapped: Dict[int, tuple] = {}  # address -> (handle, bytes)
        self._free: Dict[int, List] = {}  # chunk bytes -> unmapped handles
        self._num_created = 0

    def creates(self) -> int:
        return self._num_created

    def pooled(self, nbytes: int) -> int:
        return len(self._free.get(int(nbytes), ()))

    def _create(self, nbytes: int):
        handle = check_drv(cuda.cuMemCreate(nbytes, self._prop, 0), "cuMemCreate")
        self._num_created += 1
        return handle

    def reserve_pool(self, device_id: int, nbytes: int, count: int) -> None:
        """Create `count` chunks now, to be mapped later without cuMemCreate."""
        free = self._free.setdefault(int(nbytes), [])
        for _ in range(int(count)):
            free.append(self._create(int(nbytes)))

    def map_handle(self, address: int, nbytes: int, handle) -> None:
        check_drv(cuda.cuMemMap(address, nbytes, 0, handle, 0), "cuMemMap")
        check_drv(
            cuda.cuMemSetAccess(address, nbytes, self._access, len(self._access)),
            "cuMemSetAccess",
        )

    def map_range(self, address: int, nbytes: int) -> None:
        address, nbytes = int(address), int(nbytes)
        free = self._free.get(nbytes)
        handle = free.pop() if free else self._create(nbytes)
        self.map_handle(address, nbytes, handle)
        self._mapped[address] = (handle, nbytes)

    def unmap_range(self, address: int) -> None:
        handle, nbytes = self._mapped.pop(int(address))
        check_drv(cuda.cuMemUnmap(int(address), nbytes), "cuMemUnmap")
        self._free.setdefault(nbytes, []).append(handle)


@functools.cache
def _device_chunk_pool(device_id: int) -> VmmChunkPool:
    """One pool per device: a chunk the KV band gives up is the chunk the index keys map next."""
    return VmmChunkPool(device_id)


def _host_numa_id(device_id: int) -> int:
    device = check_drv(cuda.cuDeviceGet(device_id), "cuDeviceGet")
    return int(
        check_drv(
            cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device
            ),
            "cuDeviceGetAttribute(HOST_NUMA_ID)",
        )
    )


def _host_numa_prop(numa_id: int):
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
    prop.location.id = int(numa_id)
    return prop


class ElasticFamily:
    """`layers` per-layer regions of `max_bytes` each in one VA reservation.

    Bytes [0, perm) of every layer are mapped at once and never unmapped; above that, chunks of the device's
    allocation granularity are mapped and unmapped together in every layer. With `host_rows`, each layer's
    region is followed by that many rows of pinned host memory, starting at row index `h0` of the layer.
    """

    def __init__(
        self,
        device,
        layers: int,
        max_bytes: int,
        perm_bytes: int,
        name: str,
        host_rows: int = 0,
        item: int = 0,
        page: int = 0,
    ):
        dev = torch.device(device)
        self.dev = dev.index if dev.index is not None else torch.cuda.current_device()
        with torch.cuda.device(self.dev):
            self.x = _device_chunk_pool(self.dev)
            self.g = get_device_granularity(self.dev)
            self.layers, self.name = layers, name
            # bytes of a layer that chunks of the pool can back
            self.cap = align_up(max_bytes, self.g)
            self.stride = self.cap
            self.host_rows = 0
            if host_rows:
                numa_id = _host_numa_id(self.dev)
                host_prop = _host_numa_prop(numa_id)
                host_granularity = get_allocation_granularity(host_prop)
                if self.g % host_granularity:
                    raise RuntimeError(
                        f"host granularity {host_granularity} does not divide the "
                        f"device granularity {self.g}"
                    )
                # The same page-aligned host row offset in every layer.
                self.h0 = align_up(-(-self.cap // item), page)
                self.host_off = self.h0 * item
                self.host_bytes = align_up(
                    self.host_off - self.cap + host_rows * item + page * item, self.g
                )
                self.stride = self.cap + self.host_bytes
                self.host_rows, self.item, self.page = host_rows, item, page
            self.base = int(
                check_drv(
                    cuda.cuMemAddressReserve(
                        self.stride * max(1, layers), self.g, 0, 0
                    ),
                    "cuMemAddressReserve",
                )
            )
            if self.host_rows:
                # Never unmapped and never pooled.
                self._host_handles = []
                for layer in range(layers):
                    handle = check_drv(
                        cuda.cuMemCreate(self.host_bytes, host_prop, 0),
                        "cuMemCreate(host)",
                    )
                    self.x.map_handle(
                        self.base + layer * self.stride + self.cap,
                        self.host_bytes,
                        handle,
                    )
                    self._host_handles.append(handle)
                logger.info(
                    "HiSparse address range %s: %d layers x (%.2f GiB device span + "
                    "%.2f GiB host rows on NUMA node %d); host row 0 at index %d",
                    name,
                    layers,
                    self.cap / 2**30,
                    self.host_bytes / 2**30,
                    numa_id,
                    self.h0,
                )
            self.perm = align_up(perm_bytes, self.g) if perm_bytes > 0 else 0
            if self.perm:
                for layer in range(layers):
                    self.x.map_range(self.base + layer * self.stride, self.perm)
        self.mapped = self.perm  # bytes mapped per layer (a multiple of g)

    def view(self, layer: int, nbytes: int) -> torch.Tensor:
        assert nbytes <= self.stride
        return tensor_from_pointer(
            self.base + layer * self.stride, nbytes, device_id=self.dev
        )

    @property
    def layer_num(self) -> int:
        return self.layers

    def host_row0(self, layer: int) -> int:
        return self.h0

    def unified_view(self, layer: int) -> torch.Tensor:
        """The layer's index space as uint8: device rows, band, a gap, then its host rows."""
        return self.view(layer, (self.h0 + self.host_rows) * self.item)

    def host_region(self) -> torch.Tensor:
        """All layers' host rows as uint8 [layers, host_rows * item], one slab per layer."""
        whole = tensor_from_pointer(
            self.base, self.stride * self.layers, device_id=self.dev
        ).view(self.layers, self.stride)
        return whole[:, self.host_off : self.host_off + self.host_rows * self.item]

    def handles(self) -> int:
        """Chunks mapped above the permanent region (the part counted against the budget)."""
        return (self.mapped - self.perm) // self.g * self.layers

    def chunk_bytes(self) -> int:
        return self.g * self.layers

    def grow_to(self, nbytes: int) -> int:
        """Map chunks until [0, nbytes) of every layer is backed; returns chunks added per layer."""
        target = min(self.cap, align_up(nbytes, self.g))
        added = 0
        with torch.cuda.device(self.dev):
            while self.mapped < target:
                for layer in range(self.layers):
                    self.x.map_range(
                        self.base + layer * self.stride + self.mapped, self.g
                    )
                self.mapped += self.g
                added += 1
        return added

    def shrink_to(self, nbytes: int) -> int:
        """Unmap whole chunks above max(nbytes, perm). The caller guarantees that nothing there is in
        use and that the device is idle on that memory. Returns chunks removed per layer."""
        target = max(self.perm, align_up(nbytes, self.g))
        removed = 0
        with torch.cuda.device(self.dev):
            while self.mapped > target:
                self.mapped -= self.g
                for layer in range(self.layers):
                    self.x.unmap_range(self.base + layer * self.stride + self.mapped)
                removed += 1
        return removed
