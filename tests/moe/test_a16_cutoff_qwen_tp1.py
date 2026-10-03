"""A16 token cutoff on ModelOpt NVFP4 MoE at the Qwen3.8 TP=1 expert shape (K=2560, I=640, top-10).

Serving (vLLM VLLM_B12X_A16_MAX_TOKENS) plans the experts in A4 mode with a16_max_tokens set, so
calls of at most the cutoff run the W4A16 backend on the NVFP4 weights and larger calls stay A4.
These tests run that plan at M=5..64 through the prepared route-pack and direct launches, eagerly
and under CUDA graph replay with changed inputs, against an FP32 dequantized reference.
"""

from __future__ import annotations

import pytest
import torch

from b12x.moe import fused_moe
from b12x.preparation import PreparationSession

E, K, I, TOPK, CUTOFF = 32, 2560, 640, 10, 40
COUNTS = (5, 10, 20, 40, 64)


def _weights():
    from .test_w4a16_e2e import _quantize_dense_moe_weight_storage

    torch.manual_seed(20261002)
    g13 = torch.ones(E, dtype=torch.float32, device="cuda")
    g2 = torch.ones(E, dtype=torch.float32, device="cuda")
    w13, s13 = _quantize_dense_moe_weight_storage(torch.randn(E, 2 * I, K, device="cuda") * 0.05, g13)
    w2, s2 = _quantize_dense_moe_weight_storage(torch.randn(E, K, I, device="cuda") * 0.05, g2)
    return w13, s13, g13, w2, s2, g2


def _experts(raw):
    w13, s13, g13, w2, s2, g2 = raw
    ones = torch.ones(E, device="cuda")
    plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=fused_moe.ActivationSpec(
            mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16, a16_max_tokens=CUTOFF,
        ),
        geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=K, intermediate_size=I),
    )
    return fused_moe.prepare_weights(plan=plan, weights=fused_moe.PackedWeights(
        w13=w13, w2=w2, w13_block_scales=s13, w2_block_scales=s2,
        w13_global_scales=g13, w2_global_scales=g2, input_scale=ones, intermediate_scale=ones,
    ))


def _inputs(rows):
    x = (torch.randn(rows, K, device="cuda") * 0.25).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(rows)]).to(torch.int32)
    weights = torch.softmax(torch.randn(rows, TOPK, device="cuda"), dim=-1)
    return x, ids, weights


def _reference(raw, x, ids, weights):
    from b12x.moe._shared.kernels.reference import moe_reference_w4a16_f32

    return moe_reference_w4a16_f32(x, *raw, ids, weights, E, K, I)


def _cos(actual, expected):
    return torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0,
    ).item()


def _scratch(state):
    return tuple(torch.empty(s.shape, dtype=s.dtype, device=s.device) for s in state.scratch.scratch_specs())


def _check_rows(raw, execution, rows, x, ids, weights, floor):
    """Eager run, then graph capture and three replays (inputs changed before the second)."""
    # unlisted rows at or below the cutoff bind the A16 capacity variant at the cutoff
    variant = rows if rows in execution.variants else (
        CUTOFF if rows <= CUTOFF else min(c for c in execution.variants if c >= rows))
    scratch = _scratch(execution.variants[variant].prepared.state)
    a, topk_ids = x[:rows].clone(), ids[:rows].clone()
    out = torch.empty_like(a)
    binding = fused_moe.bind(execution, scratch=scratch, a=a, topk_ids=topk_ids,
                             topk_weights=weights[:rows], output=out)
    expected = _reference(raw, a, topk_ids, weights[:rows])
    actual = fused_moe.run(binding=binding)
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual), rows
    assert _cos(actual, expected) > floor, (rows, _cos(actual, expected))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_moe.run(binding=binding)
    for replay in range(3):
        if replay == 1:
            a.neg_()
            topk_ids.add_(1).remainder_(E)
            expected = _reference(raw, a, topk_ids, weights[:rows])
        actual.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(actual).all(), (rows, replay)
        assert _cos(actual, expected) > floor, (rows, replay, _cos(actual, expected))
    graph.reset()


def _prepare(session, execution, counts, x, ids, weights, check=None):
    from b12x.preparation import PreparedCall

    def make_call(m):
        def call(state):
            scratch = _scratch(state)
            binding = state.bind(scratch=scratch, a=x[:m], topk_ids=ids[:m],
                                 topk_weights=weights[:m], output=torch.empty_like(x[:m]))
            if check is not None:
                check(m, state)
            return PreparedCall(run=lambda: state.run(binding), owners=scratch)
        return call

    session.prepare((execution.request(name="a16-cutoff", prepare_calls={m: make_call(m) for m in counts}),))
    session.freeze()


@pytest.mark.parametrize("autotune", (False, True))
def test_a16_cutoff_matches_fp32_reference_from_5_to_64_rows(tmp_path, autotune):
    from tests._reference.helpers import require_b12x

    require_b12x()
    raw = _weights()
    experts = _experts(raw)
    x, ids, weights = _inputs(max(COUNTS))
    execution = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=max(COUNTS), top_k=TOPK, warmup_token_counts=COUNTS),
    )
    assert execution.token_counts == COUNTS

    def check(m, state):
        assert (state.config.backend == "w4a16") is (m <= CUTOFF), (m, state.config)

    with PreparationSession(device=x.device, autotune=autotune, compile_workers=2, cache_dir=tmp_path) as session:
        _prepare(session, execution, COUNTS, x, ids, weights, check)
        for rows in (5, 7, 10, 13, 20, 24, 33, 40, 48, 64):
            # W4A16 rows keep BF16 activations: close to the FP32 reference. A4 rows quantize them.
            _check_rows(raw, execution, rows, x, ids, weights, 0.999 if rows <= CUTOFF else 0.97)


@pytest.mark.parametrize("route_mode", ("direct", "packed"))
def test_a16_forced_route_mode_matches_fp32_reference(tmp_path, route_mode):
    from tests._reference.helpers import require_b12x

    require_b12x()
    raw = _weights()
    experts = _experts(raw)
    # direct routing is planned only for exact small counts (<= 8 rows); packed covers 5..40
    counts = (5, 8) if route_mode == "direct" else (5, 10, 20, 40)
    x, ids, weights = _inputs(max(counts))
    execution = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=max(counts), top_k=TOPK, warmup_token_counts=counts),
        override=fused_moe.MoeDecodeConfig(
            backend="w4a16", route_planner="internal", max_active_clusters=None,
            w4a16_route_mode=route_mode,
        ),
    )

    def check(m, state):
        assert state.config.backend == "w4a16" and state.config.w4a16_route_mode == route_mode, state.config

    with PreparationSession(device=x.device, autotune=False, compile_workers=2, cache_dir=tmp_path) as session:
        _prepare(session, execution, counts, x, ids, weights, check)
        for rows in counts if route_mode == "direct" else (5, 8, 10, 13, 20, 33, 40):
            _check_rows(raw, execution, rows, x, ids, weights, 0.999)
