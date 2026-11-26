import torch

from sglang.srt.mem_cache.hisparse_shard import HiSparseShardView


class HiSparseDevicePoolMappingMixin:
    # This GPU's share of a context split over several GPUs; the default is no split.
    hs_shard = HiSparseShardView()

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def _translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self._translate_loc_to_hisparse_device(compressed_indices)

    def translate_loc_from_full_to_hisparse_device(self, full_indices: torch.Tensor):
        return self._translate_loc_to_hisparse_device(
            self.translate_loc_from_full_to_compressed(full_indices)
        )
