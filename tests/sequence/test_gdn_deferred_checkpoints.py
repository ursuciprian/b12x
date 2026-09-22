"""Deferred GDN checkpoints: contract on CPU, replay numerics on SM120.

The claim under test is narrow and exact. With ``Caps.deferred_checkpoints``
the verify pass persists one full checkpoint (column zero) plus a compact
per-token record in the speculative columns, and the accepted-prefix state is
reconstructed from those records. That reconstruction must be **bit-identical**
to the checkpoint the shipped path writes into column ``accepted - 1``, so
every comparison below is kernel against kernel at ``rtol=0, atol=0``. A torch
reference reduces in a different order and cannot settle a bit-identity claim.
"""

from __future__ import annotations

import pytest
import torch

from b12x.sequence import gdn_decode as gdn
from b12x.sequence.gdn_decode._impl import Caps
from b12x.sequence.gdn_decode._tuning import TUNING, GdnQuery

from ..conftest import require_b12x as require_sm120
from .test_gdn_decode import (  # noqa: F401  (imported fixture)
    _make_case,
    _prepared_case_lifetime,
)


_INPUTS = (
    "mixed_qkv",
    "a",
    "b",
    "z",
    "A_log",
    "dt_bias",
    "norm_weight",
    "recurrent_state",
)


def _query(**overrides) -> GdnQuery:
    fields = dict(
        gate_activation="sigmoid",
        qk_l2norm=True,
        state_dtype="float32",
        key_heads=8,
        value_heads=24,
        max_seqs=16,
        max_tokens=80,
        state_index_columns=5,
        a_log_dtype="float32",
        dt_bias_dtype="bfloat16",
    )
    fields.update(overrides)
    return GdnQuery(**fields)


# --------------------------------------------------------------------------
# Contract, CPU only. These are the regression guard for "turning the knob on
# does not invalidate the persisted plan selection on dgx-01".
# --------------------------------------------------------------------------


def test_the_knob_stays_out_of_the_selection_key() -> None:
    off = _query()
    on = _query(deferred_checkpoints=True)
    assert off.deferred_checkpoints is False
    assert "deferred_checkpoints" not in TUNING.query_fields
    assert TUNING.encode_query(off) == TUNING.encode_query(on)
    # Nothing that feeds the choice digest may move for a default-off knob.
    assert TUNING.query_schema_version == 4
    assert TUNING.config_schema_version == 4
    assert TUNING.candidate_contract_version == 3
    assert TUNING.config_fields == frozenset({"backend", "recurrent_block_v"})


def test_the_query_round_trips_the_knob() -> None:
    on = _query(deferred_checkpoints=True)
    assert GdnQuery(**on.to_dict()).deferred_checkpoints is True
    assert GdnQuery(**_query().to_dict()).deferred_checkpoints is False


@pytest.mark.parametrize(
    "overrides",
    (
        {"state_dtype": "bfloat16"},
        {"null_state_index": 0},
        {"key_heads": 24},
    ),
)
def test_unsupported_geometry_is_rejected(overrides) -> None:
    TUNING.validate_query(_query(deferred_checkpoints=True), None)
    query = _query(deferred_checkpoints=True, **overrides)
    with pytest.raises(ValueError, match="deferred GDN checkpoints require"):
        TUNING.validate_query(query, None)


def test_caps_reject_unsupported_deferred_geometry() -> None:
    device = torch.device("cuda", 0)
    base = dict(
        device=device,
        max_tokens=80,
        max_seqs=16,
        max_state_slots=128,
        key_heads=8,
        value_heads=24,
        state_index_columns=5,
        gate_activation="sigmoid",
        deferred_checkpoints=True,
    )
    assert Caps(**base).deferred_checkpoints is True
    with pytest.raises(ValueError, match="FP32 recurrent state"):
        Caps(**{**base, "state_dtype": torch.bfloat16})
    with pytest.raises(ValueError, match="null_state_index"):
        Caps(**{**base, "null_state_index": 0})
    with pytest.raises(ValueError, match="Qwen GDN only"):
        Caps(**{**base, "key_heads": 24})
    with pytest.raises(TypeError, match="must be boolean"):
        Caps(**{**base, "deferred_checkpoints": 1})


# --------------------------------------------------------------------------
# Replay numerics, SM120 only.
# --------------------------------------------------------------------------


def _paired_cases(device, live_seqs: int):
    """Two plans over identical inputs: shipped path and deferred path."""
    shape = dict(
        device=device,
        query_lengths=(5,) * live_seqs,
        max_seqs=live_seqs,
        max_tokens=5 * live_seqs,
        columns=5,
        activation="sigmoid",
        state_dtype=torch.float32,
        a_log_dtype=torch.float32,
    )
    off_binding, off = _make_case(**shape)
    on_binding, on = _make_case(**shape, deferred_checkpoints=True)
    for name in _INPUTS:
        off[name].copy_(on[name])
    for name in ("query_start_loc", "num_seqs", "num_tokens"):
        assert torch.equal(off[name], on[name])
    assert torch.equal(off["state_indices"], on["state_indices"])
    return off_binding, off, on_binding, on


@pytest.mark.parametrize("live_seqs", (1, 8, 16))
def test_commit_reproduces_every_accepted_prefix_checkpoint(live_seqs) -> None:
    device = require_sm120()
    off_binding, off, on_binding, on = _paired_cases(device, live_seqs)

    # One verify step from the same initial pool, with accepted=1 so both
    # kernels read the same column.
    gdn.run(off_binding)
    gdn.run(on_binding)
    torch.cuda.synchronize(device)
    checkpoints = off["recurrent_state"].clone()
    primed = on["recurrent_state"].clone()
    indices = on["state_indices"].tolist()
    destination = on["state_indices"][:, 0].clone().contiguous()

    for accepted in range(1, 6):
        on["recurrent_state"].copy_(primed)
        on["num_accepted_tokens"][:live_seqs] = accepted
        gdn.commit_deferred_checkpoints(on_binding, destination)
        torch.cuda.synchronize(device)
        for request in range(live_seqs):
            torch.testing.assert_close(
                on["recurrent_state"][indices[request][0]],
                checkpoints[indices[request][accepted - 1]],
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("live_seqs", (1, 8, 16))
def test_the_next_step_replays_the_accepted_prefix_in_place(live_seqs) -> None:
    device = require_sm120()
    off_binding, off, on_binding, on = _paired_cases(device, live_seqs)

    gdn.run(off_binding)
    gdn.run(on_binding)
    torch.cuda.synchronize(device)
    off_primed = off["recurrent_state"].clone()
    on_primed = on["recurrent_state"].clone()

    # Fresh activations for the second step, identical on both sides.
    for name in ("mixed_qkv", "a", "b", "z"):
        on[name].normal_(0.0, 0.25)
        off[name].copy_(on[name])
    indices = on["state_indices"].tolist()

    for accepted in range(1, 6):
        off["recurrent_state"].copy_(off_primed)
        on["recurrent_state"].copy_(on_primed)
        off["num_accepted_tokens"][:live_seqs] = accepted
        on["num_accepted_tokens"][:live_seqs] = accepted
        expected = gdn.run(off_binding).clone()
        actual = gdn.run(on_binding)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for request in range(live_seqs):
            base = indices[request][0]
            torch.testing.assert_close(
                on["recurrent_state"][base],
                off["recurrent_state"][base],
                rtol=0,
                atol=0,
            )


def test_a_single_column_plan_needs_no_records() -> None:
    device = require_sm120()
    shape = dict(
        device=device,
        query_lengths=(1, 1, 1, 1),
        max_seqs=4,
        max_tokens=4,
        columns=1,
        activation="sigmoid",
        state_dtype=torch.float32,
        a_log_dtype=torch.float32,
    )
    off_binding, off = _make_case(**shape)
    on_binding, on = _make_case(**shape, deferred_checkpoints=True)
    for name in _INPUTS:
        off[name].copy_(on[name])
    expected = gdn.run(off_binding).clone()
    actual = gdn.run(on_binding)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        on["recurrent_state"], off["recurrent_state"], rtol=0, atol=0
    )


def test_the_commit_requires_the_deferred_contract() -> None:
    device = require_sm120()
    binding, tensors = _make_case(
        device=device, query_lengths=(1,), columns=1, max_seqs=4, max_tokens=4
    )
    destination = torch.zeros(
        (int(tensors["state_indices"].shape[0]),),
        dtype=torch.int32,
        device=device,
    )
    with pytest.raises(ValueError, match="deferred_checkpoints=True"):
        gdn.commit_deferred_checkpoints(binding, destination)
