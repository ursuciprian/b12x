"""Large-row MXFP8 capacity regime of plan_regimes (CPU only: host lowering and regime routing, no kernel).

A wide-N, long-K MXFP8 capacity declared for >= LARGE_M_ROWS rows lowers to 128x128/BK64 without the large-M unroll on
low-SM parts and gains a capacity regime of LARGE_M_ROWS - 1 rows that keeps the previous plan for every live M below
LARGE_M_ROWS. Every other declaration lowers exactly as before.
"""
from types import SimpleNamespace

import pytest
import torch

from b12x.gemm import blockscaled
from b12x.gemm._preparation import _default_lowering
from b12x.gemm._tuning import DenseGemmQuery
from b12x.gemm.blockscaled._preparation import LARGE_M_ROWS, _PackedRegimeState, _dense_lowering
from b12x.gemm.blockscaled._tuning import BlockscaledConfig
from b12x.preparation import DeviceIdentity

GB10 = DeviceIdentity(vendor="nvidia", compute_capability=(12, 1), sm_count=48, product_name="NVIDIA GB10")
RTX = DeviceIdentity(vendor="nvidia", compute_capability=(12, 0), sm_count=188, product_name="RTX PRO 6000 Blackwell")
QUANTIZED = BlockscaledConfig(mode="quantized")


def _query(rows, n=2560, k=6144, expected_m=None):
    return blockscaled.BlockscaledQuery(
        recipe="mxfp8", num_tokens=rows, in_features=k, padded_in_features=k, out_features=n,
        workspace_form="provided", expected_m=expected_m,
    )


def _lower(query, identity=GB10):
    return _dense_lowering(query, QUANTIZED, SimpleNamespace(identity=identity)).to_dict()


def _previous(query, identity=GB10):
    """The lowering before this regime existed: the same dense declaration without launch overrides."""
    return _default_lowering(DenseGemmQuery(
        recipe="mxfp8", entry_point="gemm.blockscaled.mm", weight_storage="native", output_dtype="bfloat16",
        batch=1, max_rows=query.num_tokens, in_features=query.padded_in_features,
        out_features=query.out_features, output_mode="provided", alpha_mode="tensor", expected_m=query.expected_m,
    ), identity).to_dict()


def _without_rows(lowering):
    return {key: value for key, value in lowering.items() if key != "m"}


def test_large_capacity_lowers_to_the_bk64_tile_without_unroll():
    for rows in (LARGE_M_ROWS, 8192, 9072):
        p = _lower(_query(rows))
        assert (tuple(p["mma_tiler_mn"]), p["tile_k"], p["policy"]["large_m_unroll"]) == ((128, 128), 64, False)
    p = _lower(_query(8192, n=4096, k=4096))
    assert (tuple(p["mma_tiler_mn"]), p["tile_k"]) == ((128, 128), 64)


def test_small_capacity_regime_is_the_previous_plan():
    small = _lower(_query(LARGE_M_ROWS - 1))
    assert (tuple(small["mma_tiler_mn"]), small["tile_k"], small["policy"]["large_m_unroll"]) == ((64, 128), 128, True)
    # identical to the previous 8192-row capacity plan in everything but the declared row bound
    assert _without_rows(small) == _without_rows(_previous(_query(8192)))


@pytest.mark.parametrize("query, identity", (
    (_query(8192, n=16384, k=2560), GB10),       # short K (GDN in_proj)
    (_query(8192, n=2560, k=3072), GB10),        # TP=2 out_proj/o_proj
    (_query(8192, n=336, k=10240), GB10),        # narrow N
    (_query(8192, n=1280, k=2560), GB10),        # shared expert
    (_query(3024), GB10),                        # small capacity
    (_query(8192, expected_m=6048), GB10),       # declared exact M
    (_query(8192), RTX),                         # high-SM part
))
def test_other_declarations_lower_as_before(query, identity):
    assert _lower(query, identity) == _previous(query, identity)


def test_plan_regimes_adds_the_small_capacity_only_for_large_wide_long_k():
    plan = blockscaled.plan_regimes(_query(8192), exact_m=(1, 2, 4, 8))
    assert plan.token_counts == (1, 2, 4, 8, LARGE_M_ROWS - 1, 8192)
    assert dict(plan.capacity_metadata) == {"max_rows": 8192, "exact_m": (1, 2, 4, 8)}
    assert blockscaled.plan_regimes(_query(8192, n=16384, k=2560), exact_m=(1, 2)).token_counts == (1, 2, 8192)
    assert blockscaled.plan_regimes(_query(2048), exact_m=(1, 2)).token_counts == (1, 2, 2048)
    with pytest.raises(ValueError, match="collides"):
        blockscaled.plan_regimes(_query(8192), exact_m=(1, LARGE_M_ROWS - 1))


def test_regime_routing_by_live_rows():
    k = 6144
    state = lambda rows: SimpleNamespace(query=SimpleNamespace(num_tokens=rows, in_features=k), required_workspace=rows)
    exact = {rows: state(rows) for rows in (1, 2, 4, 8)}
    regimes = _PackedRegimeState(state(8192), exact, state(LARGE_M_ROWS - 1))
    pick = lambda rows: regimes.resolve(torch.empty(rows, k, device="meta"))
    assert pick(4) is exact[4]
    for rows in (3, 9, 1264, 3024, LARGE_M_ROWS - 1):
        assert pick(rows) is regimes.small
    for rows in (LARGE_M_ROWS, 6048, 8192):
        assert pick(rows) is regimes.capacity
    with pytest.raises(ValueError):
        pick(8193)
    assert regimes.required_workspace == 8192
    plain = _PackedRegimeState(state(8192), exact)
    assert plain.resolve(torch.empty(3024, k, device="meta")) is plain.capacity
