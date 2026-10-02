#!/usr/bin/env python3
"""Expert-weight bandwidth of the NVFP4 MoE decode kernels at Qwen3.8 TP2 shapes.

One full layer bank is allocated: 512 experts x 1.3824 MB = 708 MB of FP4 payload
plus scales. Each timed call routes its M tokens to D distinct experts taken from a
window that advances by D experts per call, so consecutive calls touch disjoint
experts. A timed graph replays enough calls to cycle through the whole bank, which
keeps expert weights from being served out of L2. The benchmark refuses to run if
the rotation covers less than 512 MB.

    python benchmarks/benchmark_moe_wm.py [--backends dynamic,wm] [--shapes 5:33,20:96]

The report gives us/call and expert-weight GB/s = D x 1.3824 MB / t per
(backend, M, D). The default shapes are the measured serving distinct-expert
counts: 33 at c1 (M=5) and 96 at c4 (M=20).
"""

from __future__ import annotations

import argparse
import os

import torch

from b12x._lib.intrinsics import swizzle_block_scale
from b12x.moe import fused_moe
from b12x.preparation import PreparationSession, PreparedCall

E, K, I, TOPK = 512, 2560, 320, 10
EXPERT_BYTES = 2 * I * K // 2 + 2 * I * K // 16 + K * I // 2 + K * I // 16  # 1,382,400
MIN_ROTATION_BYTES = 512 << 20


def set_intermediate(width):
    global I, EXPERT_BYTES
    I = width
    EXPERT_BYTES = 2 * I * K // 2 + 2 * I * K // 16 + K * I // 2 + K * I // 16


def _experts(device):
    gen = torch.Generator(device=device).manual_seed(1)
    w1 = torch.randint(0, 256, (E, 2 * I, K // 2), dtype=torch.uint8, device=device, generator=gen)
    w2 = torch.randint(0, 256, (E, K, I // 2), dtype=torch.uint8, device=device, generator=gen)

    def scales(shape):
        exp = torch.randint(-9, -4, shape, device=device, generator=gen).float()
        return swizzle_block_scale(torch.exp2(exp).to(torch.float8_e4m3fn)).contiguous()

    a1 = torch.full((1,), 48.0, device=device)
    a2 = torch.full((1,), 384.0, device=device)
    plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=fused_moe.ActivationSpec(mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16),
        geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=K, intermediate_size=I),
    )
    return fused_moe.prepare_weights(plan=plan, weights=fused_moe.PackedWeights(
        w13=w1, w2=w2, w13_block_scales=scales((E, 2 * I, K // 16)),
        w2_block_scales=scales((E, K, I // 16)),
        w13_global_scales=torch.full((E,), 0.6, device=device),
        w2_global_scales=torch.full((E,), 0.7, device=device),
        input_scale=a1, intermediate_scale=a2,
    ))


def _plan(experts, m, backend, shared_input):
    if backend in ("wm", "flint"):
        os.environ["B12X_MOE_DECODE_BACKEND"] = "wm"
        override = None
    else:
        os.environ.pop("B12X_MOE_DECODE_BACKEND", None)
        override = fused_moe.MoeDecodeConfig(
            backend="dynamic", route_planner="internal", max_active_clusters=None,
            dynamic_tile_m=16, dynamic_route_mode="grouped", nvfp4_share_input=shared_input,
        )
    try:
        return fused_moe.plan_execution(
            experts=experts, capacity=fused_moe.ExecutionCapacity(max_tokens=m, top_k=TOPK),
            invocation={"fast_math": True}, override=override,
        )
    finally:
        os.environ.pop("B12X_MOE_DECODE_BACKEND", None)


def _route_sets(device, m, d):
    """Call i routes token t, slot k to expert (i*d + (t*TOPK + k) % d) % E."""
    if d > E or d < TOPK or d > m * TOPK:
        raise ValueError(f"cannot route M={m} tokens to exactly {d} distinct experts")
    calls = -(-E // d) + 1
    routes = torch.arange(m * TOPK).reshape(m, TOPK) % d
    sets = [((routes + i * d) % E).to(torch.int32).to(device).contiguous() for i in range(calls)]
    for s in sets:
        assert s.unique().numel() == d
    return sets


def _route_sets_skewed(device, m, window, skew, seed=0):
    """Real-routing stand-in: each call's tokens draw TOPK distinct experts from a window of
    ``window`` experts with Zipf(skew) popularity (shuffled per call), so per-expert row
    counts are uneven and D varies per call. Windows advance by ``window`` experts."""
    gen = torch.Generator().manual_seed(seed)
    calls = -(-E // window) + 1
    probs = 1.0 / torch.arange(1, window + 1, dtype=torch.float64) ** skew
    sets = []
    for i in range(calls):
        perm = torch.randperm(window, generator=gen)
        idx = torch.multinomial(probs.expand(m, window), TOPK, replacement=False, generator=gen)
        sets.append(((perm[idx] + i * window) % E).to(torch.int32).to(device).contiguous())
    return sets


def bench(device, experts, backend, m, d, *, replays, shared_input, skew=0.0):
    # "wm:nocompute+noscale" times a probe variant of wm (wrong results, timing only).
    # "flint" is the wm backend with B12X_MOE_WM_SCHEDULE=flint, held for plan, bind and run.
    backend, _, probe = backend.partition(":")
    os.environ["B12X_WM_TIMING_PROBE"] = probe.replace("+", ",")
    if backend == "flint":
        os.environ["B12X_MOE_WM_SCHEDULE"] = "flint"
    try:
        return _bench(device, experts, backend, m, d, replays=replays, shared_input=shared_input,
                      skew=skew, label=backend + (":" + probe if probe else ""))
    finally:
        os.environ.pop("B12X_WM_TIMING_PROBE", None)
        os.environ.pop("B12X_MOE_WM_SCHEDULE", None)


def _bench(device, experts, backend, m, d, *, replays, shared_input, skew, label):
    plan = _plan(experts, m, backend, shared_input)
    a = (torch.randn(m, K, device=device) * 0.35).to(torch.bfloat16)
    weights = torch.full((m, TOPK), 1.0 / TOPK, device=device)
    sets = _route_sets_skewed(device, m, d, skew) if skew else _route_sets(device, m, d)
    d = sum(s.unique().numel() for s in sets) / len(sets)  # mean distinct experts per call
    rotation = int(len(sets) * d * EXPERT_BYTES)
    if rotation < MIN_ROTATION_BYTES:
        raise RuntimeError(f"weight rotation {rotation >> 20} MB < 512 MB: L2 would fake bandwidth")
    scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=device) for s in plan.scratch_specs())
    out = torch.empty(m, K, dtype=torch.bfloat16, device=device)

    def factory(state):
        sc = tuple(torch.empty(s.shape, dtype=s.dtype, device=device) for s in state.scratch.scratch_specs())
        o = torch.empty_like(a)
        b = state.bind(a=a, topk_ids=sets[0], topk_weights=weights, scratch=sc, output=o,
                       input_scales_static=True)
        return PreparedCall(run=b.run, output=o, owners=(b, sc))

    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        session.prepare((plan.request(name=f"bench-{backend}-{m}", prepare_call=factory),))
        session.freeze()
        bindings = [fused_moe.bind(plan, a=a, topk_ids=s, topk_weights=weights, scratch=scratch,
                                   output=out, input_scales_static=True) for s in sets]
        for b in bindings:
            fused_moe.run(binding=b)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for b in bindings:
                fused_moe.run(binding=b)
        graph.replay()
        torch.cuda.synchronize(device)
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(replays):
            start.record()
            graph.replay()
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop) * 1e3 / len(bindings))
        graph.reset()
    samples.sort()
    t = samples[len(samples) // 2]
    return dict(backend=label, m=m, d=round(d), us=t, us_min=samples[0],
                gbps=d * EXPERT_BYTES / (t * 1e3), rotation_mb=rotation >> 20)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backends", default="dynamic,wm")
    parser.add_argument("--shapes", default="5:33,10:60,15:80,20:96,32:150",
                        help="comma list of M:D (tokens:distinct experts)")
    parser.add_argument("--replays", type=int, default=30)
    parser.add_argument("--no-shared-input", action="store_true")
    parser.add_argument("--intermediate", type=int, default=320, help="320 = TP2 rank, 640 = TP1")
    parser.add_argument("--skew", type=float, default=0.0,
                        help="Zipf exponent of skewed routing; D in --shapes is then the window")
    args = parser.parse_args()
    set_intermediate(args.intermediate)
    device = torch.device("cuda")
    experts = _experts(device)
    print(f"bank {E} experts x {EXPERT_BYTES} B = {E * EXPERT_BYTES / 1e6:.1f} MB (I={I}, skew={args.skew})")
    print(f"{'backend':26} {'M':>3} {'D':>4} {'us/call':>9} {'min':>9} {'GB/s':>7} {'rot MB':>7}")
    for shape in args.shapes.split(","):
        m, d = (int(v) for v in shape.split(":"))
        for backend in args.backends.split(","):
            r = bench(device, experts, backend, m, d, replays=args.replays,
                      shared_input=not args.no_shared_input, skew=args.skew)
            print(f"{r['backend']:26} {r['m']:>3} {r['d']:>4} {r['us']:>9.1f} {r['us_min']:>9.1f} "
                  f"{r['gbps']:>7.1f} {r['rotation_mb']:>7}", flush=True)


if __name__ == "__main__":
    main()
