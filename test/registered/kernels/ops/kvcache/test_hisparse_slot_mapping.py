"""Portable HiSparse logical-to-physical slot translation kernel checks."""

import unittest
import weakref

import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=20, stage="base-a-jit-kernel-unit", runner_config="amd")


class TestHiSparseSlotMapping(CustomTestCase):
    def test_accepts_weak_mapping_used_by_pool(self):
        from sglang.kernels.ops.kvcache.hisparse_slot_mapping import (
            translate_padded_hisparse_locations as translate,
        )

        mapping = torch.arange(32, dtype=torch.int64, device="cuda") + 100
        locations = torch.tensor([17, -1, 18], device="cuda")

        actual = translate(weakref.proxy(mapping), locations)

        torch.testing.assert_close(actual, torch.tensor([117, -1, 118], device="cuda"))

    def test_fused_slot_mapping_matches_padded_gather(self):
        from sglang.kernels.ops.kvcache.hisparse_slot_mapping import (
            translate_padded_hisparse_locations as translate,
        )

        for map_dtype in (torch.int32, torch.int64):
            mapping = torch.arange(257, dtype=map_dtype, device="cuda") * 3 + 1
            for loc_dtype in (torch.int32, torch.int64):
                for count in (0, 1, 3, 127, 129, 1024):
                    for stride in (1, 2):
                        with self.subTest(
                            map_dtype=map_dtype,
                            loc_dtype=loc_dtype,
                            count=count,
                            stride=stride,
                        ):
                            locations = (
                                torch.arange(
                                    count * stride, dtype=loc_dtype, device="cuda"
                                )[::stride]
                                % 257
                            )
                            if stride == 2:
                                storage = torch.zeros(
                                    count * 2, dtype=loc_dtype, device="cuda"
                                )
                                storage[::2] = locations
                                locations = storage[::2]
                            locations[::3] = -1
                            locations[1::7] = -2
                            original = locations.clone()
                            expected = torch.where(
                                locations >= 0,
                                mapping[locations.clamp_min(0)],
                                locations,
                            )
                            actual = translate(mapping, locations)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            torch.testing.assert_close(
                                locations, original, rtol=0, atol=0
                            )

    def test_split_slot_mapping_matches_torch(self):
        """A context split over several GPUs: widened logical slot -> widened physical
        slot, the shard kept and that shard's row mapped; against a torch reference."""
        from sglang.kernels.ops.kvcache.hisparse_slot_mapping import (
            translate_padded_hisparse_locations as translate_padded,
        )
        from sglang.kernels.ops.kvcache.hisparse_slot_mapping import (
            translate_split_hisparse_locations as translate,
        )

        generator = torch.Generator(device="cuda").manual_seed(0)
        num_rows = 517
        for shard_count in (1, 2, 4, 8):
            for map_dtype in (torch.int32, torch.int64):
                # Physical rows far from the logical ones, device rows and "host" rows alike.
                mapping = (
                    torch.randperm(num_rows, device="cuda", generator=generator) * 5 + 3
                ).to(map_dtype)
                for loc_dtype in (torch.int32, torch.int64):
                    for count in (0, 1, 3, 127, 129, 4096):
                        for stride in (1, 2):
                            with self.subTest(
                                shard_count=shard_count,
                                map_dtype=map_dtype,
                                loc_dtype=loc_dtype,
                                count=count,
                                stride=stride,
                            ):
                                storage = torch.randint(
                                    0,
                                    num_rows * shard_count,
                                    (count * stride,),
                                    device="cuda",
                                    generator=generator,
                                ).to(loc_dtype)
                                locations = storage[::stride]
                                locations[::3] = -1
                                locations[1::7] = -2
                                original = locations.clone()
                                valid = locations >= 0
                                slot = locations.clamp_min(0).to(torch.int64)
                                expected = torch.where(
                                    valid,
                                    mapping[slot // shard_count].to(torch.int64)
                                    * shard_count
                                    + slot % shard_count,
                                    locations.to(torch.int64),
                                ).to(torch.promote_types(map_dtype, loc_dtype))
                                actual = translate(
                                    mapping, locations, shard_count=shard_count
                                )
                                self.assertEqual(actual.dtype, expected.dtype)
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0
                                )
                                torch.testing.assert_close(
                                    locations, original, rtol=0, atol=0
                                )
                                # The shard survives; the row is that shard's physical row.
                                self.assertTrue(
                                    bool(
                                        (
                                            actual[valid] % shard_count
                                            == locations[valid] % shard_count
                                        ).all()
                                    )
                                )
                                if shard_count == 1:
                                    torch.testing.assert_close(
                                        actual,
                                        translate_padded(mapping, locations),
                                        rtol=0,
                                        atol=0,
                                    )

    def test_split_slot_mapping_accepts_the_pool_proxy_and_the_last_row(self):
        from sglang.kernels.ops.kvcache.hisparse_slot_mapping import (
            translate_split_hisparse_locations as translate,
        )

        mapping = torch.arange(32, dtype=torch.int64, device="cuda") + 100
        # Slots 124..127 are the four shards of the last logical row.
        locations = torch.tensor([17, -1, 18, 124, 127], device="cuda")
        actual = translate(weakref.proxy(mapping), locations, shard_count=4)
        torch.testing.assert_close(
            actual, torch.tensor([104 * 4 + 1, -1, 104 * 4 + 2, 131 * 4, 131 * 4 + 3], device="cuda")
        )


if __name__ == "__main__":
    unittest.main()
