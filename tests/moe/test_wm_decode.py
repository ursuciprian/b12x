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
- the RMS of wm minus dynamic stays under max(1.25 x dynamic's own run-to-run
  difference, 1.5 x the two kernels' combined oracle distance); on GB10 dynamic vs
  dynamic alone measures about 9e-3 at one row;
- the oracle metrics use the thresholds of the existing dynamic tests.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

from b12x._lib.intrinsics import swizzle_block_scale
from b12x.moe import fused_moe
from b12x.moe._shared.kernels.reference import compare_to_reference, moe_reference_nvfp4
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x

# B12X_WM_TEST_INTERMEDIATE=640 runs the TP=1 (full intermediate) geometry.
E, K, TOPK = 512, 2560, 10
I = int(os.environ.get("B12X_WM_TEST_INTERMEDIATE", "320"))


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


def _poison(tensors):
    """Fill with 0xFF bytes (BF16 NaN): a flint h read that no write of the same launch
    preceded turns the output NaN. Bind maps views only and run re-zeros the barrier."""
    for t in tensors:
        t.view(-1).view(torch.uint8).fill_(0xFF)


def _run(plan, device, a, ids, weights, rows, poison=False):
    scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=device) for s in plan.scratch_specs())
    if poison:
        _poison(scratch)
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
                out_w, _, _ = _run(wm, device, a, ids, weights, rows, poison=True)
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
                assert diff <= max(1.25 * noise, 1.5 * math.hypot(err_w, err_d)), ctx
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
        out_w, _, _ = _run(wm, device, a, ids, weights, capacity, poison=True)
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
    session = _prepare(device, (wm,), a, ids, weights)
    try:
        for rows in (1, 7, capacity):
            out, binding, _ = _run(wm, device, a, ids, weights, rows, poison=True)
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
                _poison((binding.packed_input,))  # flint's h rows
                allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
                ref = moe_reference_nvfp4(
                    a[:rows], w["w1"], w["w1_scale"], w["w1_alpha"], w["w2"], w["w2_scale"],
                    w["w2_alpha"], w["a1"], w["a2"], ids[:rows], weights[:rows], E, K, I,
                    quant_scale_math="dynamic_fast",
                ).float()
                assert out.isfinite().all()
                err = _rms(out.float() - ref) / _rms(ref)
                assert compare_to_reference(out.float(), ref).cos >= 0.9999 and err <= 0.015, (rows, step, err)
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


def test_wm_min_tokens_control(monkeypatch):
    """B12X_MOE_WM_MIN_TOKENS adds a floor only when set; out-of-range values fail."""
    from b12x.moe.fused_moe._preparation import _control_snapshot

    monkeypatch.setenv("B12X_MOE_DECODE_BACKEND", "wm")
    monkeypatch.setenv("B12X_MOE_WM_MAX_TOKENS", "28")
    monkeypatch.delenv("B12X_MOE_WM_MIN_TOKENS", raising=False)
    assert "wm_min_tokens" not in _control_snapshot().to_dict()
    monkeypatch.setenv("B12X_MOE_WM_MIN_TOKENS", "5")
    assert _control_snapshot().to_dict()["wm_min_tokens"] == 5
    monkeypatch.setenv("B12X_MOE_WM_MIN_TOKENS", "29")
    with pytest.raises(ValueError):
        _control_snapshot()


# ---------------------------------------------------------------------------
# flint-specific GPU tests (B3/M3 interleave; deferred extremes from the
# review's item 5). These force B12X_MOE_WM_SCHEDULE=flint themselves rather
# than relying on flint_job.sh's outer env wrapping, so they exercise flint
# correctly even when this file is run standalone.


def test_flint_dynamic_interleave_shared_workspace(monkeypatch):
    """M3/B3: dynamic and flint launches, back-to-back on one physical
    workspace (the realistic M>32-fallback pattern), must not corrupt either
    backend's output. Exercises B2's barrier reset directly: whichever plan
    ran last left its own state behind on the shared scratch, and the next
    flint call must still be correct."""
    device = require_b12x()
    monkeypatch.setenv("B12X_MOE_WM_SCHEDULE", "flint")
    w = _weights(device, seed=71, per_expert_scales=False)
    experts = _experts(w, "w13")
    capacity = 20
    a = (torch.randn(capacity, K, device=device) * 0.35).to(torch.bfloat16)
    dyn = _plan(experts, capacity, "dynamic", monkeypatch)
    flint = _plan(experts, capacity, "wm", monkeypatch)
    ids0, weights0 = _routes(device, capacity, "spread", seed=1)
    session = _prepare(device, (dyn, flint), a, ids0, weights0)
    try:
        dyn_specs = dyn.scratch_specs()
        flint_specs = flint.scratch_specs()
        assert [(s.shape, s.dtype) for s in dyn_specs] == [
            (s.shape, s.dtype) for s in flint_specs
        ], "dynamic and flint scratch layouts diverged; the shared-workspace scenario no longer applies"
        shared_scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device) for s in dyn_specs
        )

        def run_on_shared(plan, ids, weights):
            out = torch.full((capacity, K), float("nan"), dtype=torch.bfloat16, device=device)
            binding = fused_moe.bind(
                plan, a=a, topk_ids=ids, topk_weights=weights,
                scratch=shared_scratch, output=out, input_scales_static=True,
            )
            fused_moe.run(binding=binding)
            torch.cuda.synchronize(device)
            return out

        for round_idx, pattern in enumerate(("spread", "hot", "clustered", "disjoint")):
            ids, weights = _routes(device, capacity, pattern, seed=200 + round_idx)
            order = (dyn, flint) if round_idx % 2 == 0 else (flint, dyn)
            for plan in order:
                if plan is flint:
                    _poison(shared_scratch)
                out = run_on_shared(plan, ids, weights)
                ref = moe_reference_nvfp4(
                    a, w["w1"], w["w1_scale"], w["w1_alpha"], w["w2"], w["w2_scale"],
                    w["w2_alpha"], w["a1"], w["a2"], ids, weights, E, K, I,
                    quant_scale_math="dynamic_fast",
                ).float()
                assert out.isfinite().all(), (pattern, plan is flint)
                err = _rms(out.float() - ref) / _rms(ref)
                ctx = (pattern, "flint" if plan is flint else "dynamic", err)
                assert compare_to_reference(out.float(), ref).cos >= 0.9999 and err <= 0.015, ctx
    finally:
        session.__exit__(None, None, None)


def test_flint_barrier_reuse_repeated_eager_calls(monkeypatch):
    """100 back-to-back eager flint launches on one binding: the barrier must
    reset/self-clean correctly every time, not just on the first call."""
    device = require_b12x()
    monkeypatch.setenv("B12X_MOE_WM_SCHEDULE", "flint")
    w = _weights(device, seed=72, per_expert_scales=False)
    experts = _experts(w, "w13")
    capacity = 20
    a = (torch.randn(capacity, K, device=device) * 0.35).to(torch.bfloat16)
    ids, weights = _routes(device, capacity, "spread", seed=2)
    flint = _plan(experts, capacity, "wm", monkeypatch)
    session = _prepare(device, (flint,), a, ids, weights)
    try:
        patterns = ("spread", "clustered", "hot", "disjoint")
        for i in range(100):
            ids, weights = _routes(device, capacity, patterns[i % len(patterns)], seed=1000 + i)
            out, _, _ = _run(flint, device, a, ids, weights, capacity, poison=True)
            assert out.isfinite().all(), i
            ref = moe_reference_nvfp4(
                a, w["w1"], w["w1_scale"], w["w1_alpha"], w["w2"], w["w2_scale"],
                w["w2_alpha"], w["a1"], w["a2"], ids, weights, E, K, I,
                quant_scale_math="dynamic_fast",
            ).float()
            err = _rms(out.float() - ref) / _rms(ref)
            assert compare_to_reference(out.float(), ref).cos >= 0.9999 and err <= 0.015, (i, err)
    finally:
        session.__exit__(None, None, None)


@pytest.mark.parametrize("capacity,minimal,label", [
    (10, True, "D==TOPK (minimal distinct experts)"),
    (32, False, "D>48 (maximal distinct experts)"),
])
def test_flint_distinct_expert_extremes(capacity, minimal, label, monkeypatch):
    """flint's byte-balanced schedule must hold at both ends of D (distinct
    touched experts): the floor (every token shares the same TOPK experts)
    and well above the SM count."""
    device = require_b12x()
    monkeypatch.setenv("B12X_MOE_WM_SCHEDULE", "flint")
    w = _weights(device, seed=73 + capacity, per_expert_scales=False)
    experts = _experts(w, "w13")
    a = (torch.randn(capacity, K, device=device) * 0.35).to(torch.bfloat16)
    if minimal:
        # Every token routes to the identical TOPK experts: D == TOPK, the
        # smallest distinct-expert count flint's schedule can see.
        gen = torch.Generator(device="cpu").manual_seed(capacity)
        row = torch.randperm(E, generator=gen)[:TOPK]
        ids = row.unsqueeze(0).expand(capacity, TOPK).to(torch.int32).to(device).contiguous()
        wts = torch.rand(capacity, TOPK, generator=gen).add_(0.25)
        weights = (wts / wts.sum(dim=1, keepdim=True)).to(device).contiguous()
    else:
        ids, weights = _routes(device, capacity, "spread", seed=capacity)
    d = ids.unique().numel()
    if minimal:
        assert d == TOPK, (label, d)
    else:
        assert d > 48, (label, d, "seed did not reach the labeled distinct-expert extreme")
    flint = _plan(experts, capacity, "wm", monkeypatch)
    session = _prepare(device, (flint,), a, ids, weights)
    try:
        out, _, _ = _run(flint, device, a, ids, weights, capacity, poison=True)
        assert out.isfinite().all(), label
        ref = moe_reference_nvfp4(
            a, w["w1"], w["w1_scale"], w["w1_alpha"], w["w2"], w["w2_scale"],
            w["w2_alpha"], w["a1"], w["a2"], ids, weights, E, K, I,
            quant_scale_math="dynamic_fast",
        ).float()
        err = _rms(out.float() - ref) / _rms(ref)
        assert compare_to_reference(out.float(), ref).cos >= 0.9999 and err <= 0.015, (label, d, err)
    finally:
        session.__exit__(None, None, None)


def _oracle(w, a, ids, weights):
    return moe_reference_nvfp4(
        a, w["w1"], w["w1_scale"], w["w1_alpha"], w["w2"], w["w2_scale"],
        w["w2_alpha"], w["a1"], w["a2"], ids, weights, E, K, I,
        quant_scale_math="dynamic_fast",
    ).float()


def test_flint_matches_wm_tightly(monkeypatch):
    """flint vs wm (item schedule) on identical inputs.

    Tolerance. flint reuses wm's quantizers, QMMA, split-K order, SiLU, BF16 h rounding
    and FC2 epilogue, and both add one BF16 partial per (route, element), 10 adds per
    output element. The only difference is the order of those 10 BF16 atomics, so
    flint vs wm has the distribution of wm vs wm: two independent BF16 combine orders,
    about sqrt(2) x 3.2e-3 = 4.5e-3 relative RMS (module docstring). The checks
    require the global RMS difference under max(1.25 x wm's own run-to-run difference,
    6e-3) and every row under 1e-2. One stale 16 B h vector per route-item (about 3%
    on the affected rows) fails the row check; an unwritten h read is NaN (poison)."""
    device = require_b12x()
    w = _weights(device, seed=75, per_expert_scales=True)
    experts = _experts(w, "w13")
    capacity = 20
    a = (torch.randn(capacity, K, device=device) * 0.35).to(torch.bfloat16)
    ids0, weights0 = _routes(device, capacity, "spread", seed=7)
    cases = [(pattern, rows) for pattern in ("spread", "clustered", "hot", "disjoint")
             for rows in (1, capacity)]
    outs = {}
    for schedule in ("item", "flint"):
        monkeypatch.setenv("B12X_MOE_WM_SCHEDULE", schedule)
        plan = _plan(experts, capacity, "wm", monkeypatch)
        session = _prepare(device, (plan,), a, ids0, weights0)
        try:
            for pattern, rows in cases:
                ids, weights = _routes(device, capacity, pattern, seed=300 + len(pattern))
                outs[schedule, pattern, rows] = [
                    _run(plan, device, a, ids, weights, rows, poison=True)[0].float()
                    for _ in range(2 if schedule == "item" else 1)
                ]
        finally:
            session.__exit__(None, None, None)
    for pattern, rows in cases:
        wm1, wm2 = outs["item", pattern, rows]
        (fl,) = outs["flint", pattern, rows]
        assert fl.isfinite().all(), (pattern, rows)
        scale = max(_rms(wm1), 1e-6)
        diff = _rms(fl - wm1) / scale
        noise = _rms(wm2 - wm1) / scale
        row = ((fl - wm1).square().mean(1).sqrt() / wm1.square().mean(1).sqrt().clamp_min(1e-6)).max().item()
        ctx = (pattern, rows, diff, noise, row)
        assert diff <= max(1.25 * noise, 6e-3), ctx
        assert row <= 1e-2, ctx


def test_flint_single_graph_many_replays(monkeypatch):
    """One captured flint graph, 60 replays with live routes and activations and h
    poisoned before each: the captured barrier reset and the h relay hold every time."""
    device = require_b12x()
    monkeypatch.setenv("B12X_MOE_WM_SCHEDULE", "flint")
    w = _weights(device, seed=76, per_expert_scales=False)
    experts = _experts(w, "w13")
    capacity = 20
    a = (torch.randn(capacity, K, device=device) * 0.35).to(torch.bfloat16)
    ids, weights = _routes(device, capacity, "spread", seed=8)
    flint = _plan(experts, capacity, "wm", monkeypatch)
    session = _prepare(device, (flint,), a, ids, weights)
    try:
        out, binding, _ = _run(flint, device, a, ids, weights, capacity, poison=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused_moe.run(binding=binding)
        patterns = ("hot", "spread", "clustered", "disjoint")
        for step in range(60):
            new_ids, new_w = _routes(device, capacity, patterns[step % len(patterns)], seed=500 + step)
            ids.copy_(new_ids)
            weights.copy_(new_w)
            a.neg_()
            out.fill_(float("nan"))
            _poison((binding.packed_input,))
            graph.replay()
            torch.cuda.synchronize(device)
            assert out.isfinite().all(), step
            ref = _oracle(w, a, ids, weights)
            err = _rms(out.float() - ref) / _rms(ref)
            assert compare_to_reference(out.float(), ref).cos >= 0.9999 and err <= 0.015, (step, err)
        graph.reset()
    finally:
        session.__exit__(None, None, None)


def test_flint_standalone_pool_repeated_calls(monkeypatch):
    """flint through a standalone TPMoEWorkspacePool (no shared arena), the sglang
    path: the pool allocates and resolves one workspace under the runtime key, and 100
    calls on it stay correct with h poisoned before each. (The run path rebuilds the
    workspace with volatile_launch_state=True, so the barrier is still re-zeroed.)"""
    import b12x.moe.fused_moe._impl as tp_moe

    device = require_b12x()
    monkeypatch.setenv("B12X_MOE_WM_SCHEDULE", "flint")
    w = _weights(device, seed=77, per_expert_scales=False)
    experts = _experts(w, "w13")
    capacity = 20
    a = (torch.randn(capacity, K, device=device) * 0.35).to(torch.bfloat16)
    config = fused_moe.MoeDecodeConfig(backend="wm", route_planner="internal", max_active_clusters=None)
    pool = tp_moe.TPMoEWorkspacePool()
    out = torch.empty(capacity, K, dtype=torch.bfloat16, device=device)
    patterns = ("spread", "clustered", "hot", "disjoint")
    for i in range(100):
        ids, weights = _routes(device, capacity, patterns[i % len(patterns)], seed=2000 + i)
        binding = pool.bind_fp4(a=a, experts=experts._impl, topk_weights=weights, topk_ids=ids,
                                output=out, input_scales_static=True, fast_math=True,
                                decode_config=config)
        _poison((binding.packed_input,))
        out.fill_(float("nan"))
        binding.run()
        torch.cuda.synchronize(device)
        assert out.isfinite().all(), i
        ref = _oracle(w, a, ids, weights)
        err = _rms(out.float() - ref) / _rms(ref)
        assert compare_to_reference(out.float(), ref).cos >= 0.9999 and err <= 0.015, (i, err)
    assert len(pool.workspaces) == 1, list(pool.workspaces)
