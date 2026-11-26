"""SGLANG_HISPARSE_DFLASH_STAGING: a request that leaves HiSparse staging under DFLASH gets the
draft input a PD decode builds for an arriving request; off, DFLASH raises there as before."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers import overlap_utils
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.speculative.dflash_disaggregation import (
    build_dflash_disagg_draft_input,
)
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _future_map(*, dflash: bool):
    future_map = object.__new__(FutureMap)
    future_map.spec_algo = SimpleNamespace(
        is_dflash=lambda: dflash,
        is_eagle=lambda: not dflash,
        is_standalone=lambda: False,
    )
    future_map.flush_deferred_relay = MagicMock()
    future_map.publish = MagicMock()
    future_map.stash = MagicMock()
    future_map.dsa_topk_indices_buf = None
    return future_map


def _switch(on: bool):
    return patch.object(
        overlap_utils.envs.SGLANG_HISPARSE_DFLASH_STAGING, "get", return_value=on
    )


class TestHiSparseDflashStaging(CustomTestCase):
    def setUp(self):
        self.indices = torch.tensor([7, 2, 5])
        self.bonus = torch.tensor([11, 12, 13], dtype=torch.int32)
        self.seq_lens = torch.tensor([100, 4097, 9001])

    def _staged(self, future_map, bonus=None):
        return future_map.make_staged_spec_input(
            self.indices,
            bonus_tokens=self.bonus if bonus is None else bonus,
            seq_lens=self.seq_lens,
        )

    def _reference(self):
        """The PD decode's builder on the same batch."""
        future_map = _future_map(dflash=True)
        batch = SimpleNamespace(
            seq_lens=self.seq_lens, req_pool_indices=self.indices, enable_overlap=True
        )
        return build_dflash_disagg_draft_input(batch, self.bonus, future_map), future_map

    def _same(self, got, want) -> bool:
        return (
            isinstance(got, DFlashDraftInputV2)
            and torch.equal(got.bonus_tokens, want.bonus_tokens)
            and got.bonus_tokens.dtype == want.bonus_tokens.dtype
            and torch.equal(got.new_seq_lens, want.new_seq_lens)
            and torch.equal(got.future_indices, want.future_indices)
            and got.topk_p.shape == want.topk_p.shape
            and got.hidden_states.shape == want.hidden_states.shape
        )

    def test_off_dflash_is_refused_as_before(self):
        with _switch(False), self.assertRaisesRegex(RuntimeError, "EAGLE-family"):
            self._staged(_future_map(dflash=True))

    def test_on_equals_the_pd_arrival_input(self):
        want, reference_map = self._reference()
        future_map = _future_map(dflash=True)
        with _switch(True):
            got = self._staged(future_map)
        self.assertTrue(self._same(got, want))
        # The same publication of the lengths; the scheduler stashed the bonus tokens.
        published = future_map.publish.call_args.args
        expected = reference_map.publish.call_args.args
        self.assertTrue(torch.equal(published[0], expected[0]))
        self.assertTrue(torch.equal(published[1], expected[1]))
        stashed = reference_map.stash.call_args.args[1]
        self.assertIsInstance(stashed, RelayPayload)
        future_map.flush_deferred_relay.assert_called_once()

    def test_a_shifted_bonus_token_is_seen(self):
        """Mutant: the test's comparison must tell a wrong hand-over from a right one."""
        want, _ = self._reference()
        with _switch(True):
            got = self._staged(_future_map(dflash=True), bonus=self.bonus.roll(1))
        self.assertFalse(self._same(got, want))

    def test_eagle_keeps_its_input_with_the_switch_on(self):
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        for on in (False, True):
            future_map = _future_map(dflash=False)
            with _switch(on):
                got = self._staged(future_map)
            self.assertIsInstance(got, EagleDraftInput)
            self.assertIs(got.future_indices, self.indices)
            future_map.publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
