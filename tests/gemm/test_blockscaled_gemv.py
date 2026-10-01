"""Weight-only MXFP8 GEMV (mode "gemv") against the A16 path and an FP32 oracle."""

from __future__ import annotations

import pytest
import torch

from b12x.gemm import blockscaled
from b12x.gemm.blockscaled import _gemv, _preparation, _tuning
from b12x.gemm.blockscaled._tuning import BlockscaledConfig
from b12x._lib.runtime_control import kernel_resolution_guard
from tests._reference.helpers import require_b12x
from tests.gemm.test_blockscaled_a16 import assert_close, make_weight, prepared_execution


@pytest.fixture(autouse=True)
def gemv_enabled(monkeypatch):
    monkeypatch.setattr(_tuning, "_DENSE_GEMV", True)


def _run(source, weight, m, config, capacity=None):
    n = weight.out_features
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    query = blockscaled.query_from_call(source, weight, activation_mode="auto", out=out, expected_m=m)
    scratch = torch.full((max(1, _preparation._workspace_bytes(query, config)),), 255,
                         device="cuda", dtype=torch.uint8)
    with prepared_execution(source, weight, activation_mode="auto", out=out, workspace=scratch,
                            expected_m=m, config=config, name="gemv") as (_, plan):
        blockscaled.mm(source, weight, out=out, workspace=scratch, plan=plan)
    return out


SHAPES = [(96, 2560), (336, 10240), (10240, 320), (2560, 640), (2560, 6144), (1000, 2560), (136, 256)]


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [1, 5, 8])
@pytest.mark.parametrize("split", [1, 2, 4, 8])
def test_gemv_matches_a16_and_reference(n, k, m, split):
    require_b12x()
    if _gemv.geometry(n, k, m, split, 2, torch.cuda.get_device_properties(0).multi_processor_count) is None:
        pytest.skip("geometry infeasible")
    torch.manual_seed(n + k + m + split)
    weight, decoded, _ = make_weight("mxfp8", n, k)
    source = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    actual = _run(source, weight, m, BlockscaledConfig(mode="gemv", gemv_split=split, gemv_ctas=2))
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual)
    reference = source.float() @ decoded.T
    assert_close(actual, reference)
    a16 = _run(source, weight, m, BlockscaledConfig(mode="a16", tile_n=64, tile_k=64, split_k=1))
    # Same products, different FP32 summation order: at most a BF16 rounding step apart.
    diff = (actual.float() - a16.float()).abs()
    tol = a16.float().abs() * 2 ** -7 + reference.abs().max() * 1e-5
    assert bool((diff <= tol).all()), float((diff - tol).max())


def test_gemv_live_rows_below_capacity_and_graph_replay():
    require_b12x()
    n, k, capacity = 2560, 6144, 8
    weight, decoded, _ = make_weight("mxfp8", n, k)
    source = torch.randn(capacity, k, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(capacity, n, device="cuda", dtype=torch.bfloat16)
    config = BlockscaledConfig(mode="gemv", gemv_split=2, gemv_ctas=2)
    query = blockscaled.query_from_call(source, weight, activation_mode="auto", out=out)
    scratch = torch.empty(_preparation._workspace_bytes(query, config), device="cuda", dtype=torch.uint8)
    with prepared_execution(source, weight, activation_mode="auto", out=out, workspace=scratch,
                            config=config) as (_, plan):
        with kernel_resolution_guard("gemv live rows"):
            for m in (1, 3, 5, 8):
                out.fill_(float("nan"))
                blockscaled.mm(source[:m], weight, out=out[:m], workspace=scratch, plan=plan)
                assert_close(out[:m], source[:m].float() @ decoded.T)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                blockscaled.mm(source[:5], weight, out=out[:5], workspace=scratch, plan=plan)
            for _ in range(3):
                source.normal_()
                out.fill_(float("nan"))
                graph.replay()
                assert_close(out[:5], source[:5].float() @ decoded.T)


def test_gemv_policy_is_opt_in(monkeypatch):
    require_b12x()
    from b12x.preparation.device import detect_device
    weight, _, _ = make_weight("mxfp8", 2560, 2560)
    source = torch.randn(5, 2560, device="cuda", dtype=torch.bfloat16)
    device = detect_device(source.device).identity
    query = blockscaled.query_from_call(source, weight, expected_m=5)
    assert _tuning._default_config(query, device).mode == "gemv"
    assert "gemv" in _tuning._tuning_parameters(query, device).knobs[0].values
    monkeypatch.setattr(_tuning, "_DENSE_GEMV", False)
    query = blockscaled.query_from_call(source, weight, expected_m=5)
    assert "dense_gemv" not in query.codegen
    assert _tuning._default_config(query, device).mode == "a16"
    assert "gemv" not in _tuning._tuning_parameters(query, device).knobs[0].values
    big = torch.randn(9, 2560, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(_tuning, "_DENSE_GEMV", True)
    assert not _tuning.gemv_eligible(blockscaled.query_from_call(big, weight), device)
