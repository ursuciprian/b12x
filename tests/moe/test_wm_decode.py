"""GPU equivalence of the weight-major NVFP4 MoE decode (``wm``) with the dynamic kernel.

Qwen3.8-Flash-Next TP2 shapes (512 experts, hidden 2560, intermediate 320, top-10)
with random FP4 payloads and varied power-of-two block scales, so a wrong scale word
cannot hide behind a constant scale.

Tolerance. Both kernels quantize the same activations with the same quantizer and
run the same QMMA with FP32 accumulation over the same K order, so FC1 and the
requantized intermediate match. They differ only in how the FC2 partial of an expert
reaches the BF16 output: dynamic adds three BF16-rounded 128/128/64-channel slice
partials, and wm adds one. Each output element therefore takes 30 BF16 atomic adds
under dynamic and 10 under wm, in a nondeterministic order in both. The difference
is BF16 combine rounding. A lane-level CPU emulation of wm against the FP32-accumulating
oracle measures about 3.2e-3 relative RMS for BF16-atomic combining. Two kernels with
independent rounding can therefore differ by about sqrt(2) times that. The checks
require:

- wm is no further from the oracle than dynamic is, within 5%;
- the RMS of wm minus dynamic stays under 8e-3 of the output RMS;
- the oracle metrics use the thresholds of the existing dynamic tests.
"""

from __future__ import annotations

import os

import pytest
import torch

from b12x._lib.intrinsics import swizzle_block_scale
from b12x.moe import fused_moe
from b12x.moe._shared.kernels.reference import compare_to_reference, moe_reference_nvfp4
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x

E, K, I, TOPK = 512, 2560, 320, 10


def _weights(device, *, seed, per_expert_scales):
    gen = torch.Generator(device=device).manual_seed(seed)
    w1 = torch.randint(0, 256, (E, 2 * I, K // 2), dtype=torch.uint8, device=device, generator=gen)
    w2 = torch.randint(0, 256, (E, K, I // 2), dtype=torch.uint8, device=device, generator=gen)

    def scales(shape):
        exp = torch.randint(-9, -4, shape, device=device, generator=gen).float()
        return swizzle_block_scale(torch.exp2(exp).to(torch.float8_e4m3fn)).contiguous()

    w1_scale = scales((E, 2 * I, K // 16))
    w2_scale = scales((E, K, I // 16))
    if per_expert_scales:
        a1 = torch.linspace(32.0, 64.0, E, device=device)
        a2 = torch.linspace(256.0, 512.0, E, device=device)
    else:
        a1 = torch.full((1,), 48.0, device=device)
        a2 = torch.full((1,), 384.0, device=device)
    w1_alpha = torch.linspace(0.5, 0.8, E, device=device) / a1
    w2_alpha = torch.linspace(0.6, 0.9, E, device=device) / a2
    return dict(w1=w1, w1_scale=w1_scale, w1_alpha=w1_alpha, a1=a1,
                w2=w2, w2_scale=w2_scale, w2_alpha=w2_alpha, a2=a2)


def _experts(w, layout):
    plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout=layout),
        activation=fused_moe.ActivationSpec(mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16),
        geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=K, intermediate_size=I),
    )
    return fused_moe.prepare_weights(plan=plan, weights=fused_moe.PackedWeights(
        w13=w["w1"], w2=w["w2"], w13_block_scales=w["w1_scale"], w2_block_scales=w["w2_scale"],
        w13_global_scales=w["w1_alpha"] * w["a1"], w2_global_scales=w["w2_alpha"] * w["a2"],
        input_scale=w["a1"], intermediate_scale=w["a2"],
    ))


def _routes(device, m, pattern, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for t in range(m):
        if pattern == "spread":
            row = torch.randperm(E, generator=gen)[:TOPK]
        elif pattern == "clustered":
            row = torch.randperm(24, generator=gen)[:TOPK]
        elif pattern == "hot":  # every token shares expert 7: two passes above 16 rows
            row = torch.cat([torch.tensor([7]), 8 + torch.randperm(E - 8, generator=gen)[:TOPK - 1]])
        else:  # disjoint cyclic, the vLLM tuning corpus
            row = (torch.arange(TOPK) + t * TOPK) % E
        rows.append(row)
    ids = torch.stack(rows).to(torch.int32)
    weights = torch.rand(m, TOPK, generator=gen).add_(0.25)
    weights /= weights.sum(dim=1, keepdim=True)
    return ids.to(device).contiguous(), weights.to(device).contiguous()


def _plan(experts, capacity, backend, monkeypatch):
    if backend == "wm":
        monkeypatch.setenv("B12X_MOE_DECODE_BACKEND", "wm")
        override = None
    else:
        monkeypatch.delenv("B12X_MOE_DECODE_BACKEND", raising=False)
        override = fused_moe.MoeDecodeConfig(
            backend="dynamic", route_planner="internal", max_active_clusters=None,
            dynamic_tile_m=16, dynamic_route_mode="grouped",
        )
    plan = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=capacity, top_k=TOPK),
        invocation={"fast_math": True}, override=override,
    )
    monkeypatch.delenv("B12X_MOE_DECODE_BACKEND", raising=False)
    return plan


def _prepare(device, plans, a, ids, weights):
    def factory(state):
        scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=device)
                        for s in state.scratch.scratch_specs())
        out = torch.empty_like(a)
        binding = state.bind(a=a, topk_ids=ids, topk_weights=weights, scratch=scratch,
                             output=out, input_scales_static=True)
        return PreparedCall(run=binding.run, output=out, owners=(binding, scratch))

    session = PreparationSession(device=device, autotune=False, compile_workers=0)
    session.__enter__()
    session.prepare(tuple(p.request(name=f"wm-test-{i}", prepare_call=factory)
                          for i, p in enumerate(plans)))
    session.freeze()
    return session


def _run(plan, device, a, ids, weights, rows):
    scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=device) for s in plan.scratch_specs())
    out = torch.full((rows, K), float("nan"), dtype=torch.bfloat16, device=device)
    binding = fused_moe.bind(plan, a=a[:rows], topk_ids=ids[:rows], topk_weights=weights[:rows],
                             scratch=scratch, output=out, input_scales_static=True)
    fused_moe.run(binding=binding)
    torch.cuda.synchronize(device)
    return out, binding, scratch


def _rms(x):
    return x.float().square().mean().sqrt().item()


@pytest.mark.parametrize("capacity", [5, 20, 32])
@pytest.mark.parametrize("per_expert_scales", [False, True])
def test_wm_matches_dynamic(capacity, per_expert_scales, monkeypatch):
    device = require_b12x()
    w = _weights(device, seed=5 + capacity, per_expert_scales=per_expert_scales)
    experts = _experts(w, "w13")
    a = (torch.randn(capacity, K, device=device) * 0.35).to(torch.bfloat16)
    ids, weights = _routes(device, capacity, "spread", seed=capacity)
    dyn = _plan(experts, capacity, "dynamic", monkeypatch)
    wm = _plan(experts, capacity, "wm", monkeypatch)
    session = _prepare(device, (dyn, wm), a, ids, weights)
    try:
        report = []
        for pattern in ("spread", "clustered", "hot", "disjoint"):
            ids, weights = _routes(device, capacity, pattern, seed=17 * capacity + len(pattern))
            for rows in sorted({1, 2, 3, capacity // 2, capacity - 1, capacity} - {0}):
                out_w, _, _ = _run(wm, device, a, ids, weights, rows)
                out_d, _, _ = _run(dyn, device, a, ids, weights, rows)
                out_d2, _, _ = _run(dyn, device, a, ids, weights, rows)
                assert out_w.isfinite().all(), (pattern, rows)
                scale = max(_rms(out_d), 1e-6)
                diff = _rms(out_w.float() - out_d.float()) / scale
                noise = _rms(out_d.float() - out_d2.float()) / scale
                exact = (out_w == out_d).float().mean().item()
                ref = moe_reference_nvfp4(
                    a[:rows], w["w1"], w["w1_scale"], w["w1_alpha"], w["w2"], w["w2_scale"],
                    w["w2_alpha"], w["a1"], w["a2"], ids[:rows], weights[:rows], E, K, I,
                    quant_scale_math="dynamic_fast",
                ).float()
                err_w = _rms(out_w.float() - ref) / _rms(ref)
                err_d = _rms(out_d.float() - ref) / _rms(ref)
                cos = compare_to_reference(out_w.float(), ref).cos
                report.append((pattern, rows, exact, diff, noise, err_w, err_d))
                ctx = (pattern, rows, exact, diff, noise, err_w, err_d)
                assert cos >= 0.9999 and err_w <= 0.015, ctx
                assert err_w <= 1.05 * err_d + 1e-4, ctx
                assert diff <= 8e-3, ctx
        for line in report:
            print("wm-vs-dynamic %-9s rows=%2d exact=%.4f diff=%.2e self=%.2e "
                  "err_wm=%.2e err_dyn=%.2e" % line)
    finally:
        session.__exit__(None, None, None)


def test_wm_w31_layout_matches_dynamic(monkeypatch):
    """Serving checkpoints arrive gate-first; both kernels must read the flipped storage."""
    device = require_b12x()
    w = _weights(device, seed=91, per_expert_scales=False)
    experts = _experts(w, "w31")
    capacity = 20
    a = (torch.randn(capacity, K, device=device) * 0.35).to(torch.bfloat16)
    ids, weights = _routes(device, capacity, "spread", seed=3)
    dyn = _plan(experts, capacity, "dynamic", monkeypatch)
    wm = _plan(experts, capacity, "wm", monkeypatch)
    session = _prepare(device, (dyn, wm), a, ids, weights)
    try:
        out_w, _, _ = _run(wm, device, a, ids, weights, capacity)
        out_d, _, _ = _run(dyn, device, a, ids, weights, capacity)
        assert _rms(out_w.float() - out_d.float()) <= 8e-3 * _rms(out_d)
    finally:
        session.__exit__(None, None, None)


def test_wm_graph_replay_tracks_live_routes(monkeypatch):
    device = require_b12x()
    w = _weights(device, seed=33, per_expert_scales=False)
    experts = _experts(w, "w13")
    capacity = 20
    a = (torch.randn(capacity, K, device=device) * 0.35).to(torch.bfloat16)
    ids, weights = _routes(device, capacity, "spread", seed=4)
    wm = _plan(experts, capacity, "wm", monkeypatch)
    dyn = _plan(experts, capacity, "dynamic", monkeypatch)
    session = _prepare(device, (wm, dyn), a, ids, weights)
    try:
        for rows in (1, 7, capacity):
            out, binding, _ = _run(wm, device, a, ids, weights, rows)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused_moe.run(binding=binding)
            for step in range(3):
                new_ids, new_w = _routes(device, capacity, ("hot", "spread", "clustered")[step],
                                         seed=100 + step)
                ids.copy_(new_ids)
                weights.copy_(new_w)
                a.neg_()
                out.fill_(float("nan"))
                allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
                ref, _, _ = _run(dyn, device, a, ids, weights, rows)
                assert out.isfinite().all()
                assert _rms(out.float() - ref.float()) <= 8e-3 * _rms(ref), (rows, step)
            graph.reset()
    finally:
        session.__exit__(None, None, None)


def test_wm_env_selection_keeps_default_query(monkeypatch):
    """Unset env: identical controls (cache keys); set env: decode counts pin to wm."""
    from b12x.moe.fused_moe._preparation import _control_snapshot

    monkeypatch.delenv("B12X_MOE_DECODE_BACKEND", raising=False)
    base = _control_snapshot().to_dict()
    assert "decode_backend" not in base and "wm_max_tokens" not in base
    monkeypatch.setenv("B12X_MOE_DECODE_BACKEND", "wm")
    monkeypatch.setenv("B12X_MOE_WM_MAX_TOKENS", "20")
    pinned = _control_snapshot().to_dict()
    assert pinned["decode_backend"] == "wm" and pinned["wm_max_tokens"] == 20
    assert {k: v for k, v in pinned.items() if k not in ("decode_backend", "wm_max_tokens")} == base
    monkeypatch.setenv("B12X_MOE_DECODE_BACKEND", "bogus")
    with pytest.raises(ValueError):
        _control_snapshot()
    assert os.environ["B12X_MOE_WM_MAX_TOKENS"] == "20"
