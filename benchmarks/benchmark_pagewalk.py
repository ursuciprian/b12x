#!/usr/bin/env python3
"""Page-walk (TLB) warming microbench: NVFP4 MoE decode (dynamic) and one MXFP8 dense GEMV, TP=1 shapes.

Question: do streaming decode kernels lose bandwidth to GPU page walks when their weights are spread over
a large footprint (serving walks ~74 GB of weights per step), and does a cheap side-stream "touch one line
per 2 MiB page" kernel win it back? The touch idea comes from dgpp's WeightPrefetcher; no dgpp code is used.

Per M, per call, median over CUDA-graph replays:
  r512       weights rotate through a 512 MiB ring (one MoE bank window / 13 dense copies)
  r8g        weights rotate through an >= 8 GiB ring (7 MoE banks of 512 experts / ~200 dense copies)
  *-warm     the second of two back-to-back calls on the same weights: page walks warm, data from DRAM again
             (the per-call footprint is >= 91 MB, well over L2), derived as T(pairs) - T(singles)
  *-same     + touch of this call's pages on a low-priority side stream, forked right before the kernel
             (MoE: route-aware from topk_ids, the serving placement right after top-k)
  *-next     + touch of the next call's pages during this call (ideal prefetch)
  r512 again repeats the first config as a drift check.

The touch is one Triton kernel: per routed (token, slot) expert it loads one byte (.cg, L2 only) per 2 MiB page
of w13, w13 scales, w2 and w2 scales; for dense it walks the next weight's values and scales.

    python benchmarks/benchmark_pagewalk.py [--part moe,dense] [--shapes 5:33,20:96,40:151]
    python benchmarks/benchmark_pagewalk.py --selftest     # CPU-only checks, no GPU needed

The report ends with the expected ms/step at TP=1 c1/c4/c8 (k18 profile: 48 MoE calls per step) and the
go rule from dgpp-assessment.md: r8g >= 5% slower than r512 and the touch recovers at least half of it.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU self-test host
    triton = None

E, K, I, TOPK = 512, 2560, 640, 10
EXPERT_BYTES = 2 * I * K // 2 + 2 * I * K // 16 + K * I // 2 + K * I // 16  # 2,764,800
DENSE_N = 2 * 16 * 128 + 2 * 48 * 128  # GDN in_proj_qkvz at TP=1: q,k 16x128, v,z 48x128 -> 16384 x 2560
PAGE = 2 << 20
MOE_CALLS_PER_STEP = 48
# k18 TP=1 profile after keepalive (dgpp-assessment.md section 3): M -> (cell, MoE ms, dense ms, step ms)
PROFILE = {5: ("c1", 22.9, 24.3, 61.4), 20: ("c4", 62.1, 25.8, 111.6), 40: ("c8", 92.9, 27.0, 157.4)}


# ---------------------------------------------------------------- touch kernel and its page points

def page_points(nbytes: int, page: int = PAGE) -> list[int]:
    """Offsets one page apart from 0, ending at nbytes - 1. Consecutive points are <= page apart, so every
    page that [base, base + nbytes) intersects holds one of them, for any base alignment."""
    return [*range(0, nbytes - 1, page), nbytes - 1]


def points_table(sizes, page: int = PAGE) -> tuple[torch.Tensor, int]:
    """[len(sizes), W] int64 table of page points per region, W a power of two, short rows padded with
    their last point (a duplicate touch costs nothing)."""
    rows = [page_points(s, page) for s in sizes]
    width = 1 << (max(map(len, rows)) - 1).bit_length()
    return torch.tensor([r + [r[-1]] * (width - len(r)) for r in rows], dtype=torch.int64), width


def as_bytes(t: torch.Tensor) -> torch.Tensor:
    """uint8 view of a tensor's bytes; a non-contiguous tensor gives its whole storage."""
    if t.is_contiguous():
        return t.view(-1).view(torch.uint8)
    storage = t.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=t.device).set_(storage, 0, (storage.nbytes(),), (1,))


if triton is not None:
    @triton.jit
    def _touch_region(base, stride, pts, e, W: tl.constexpr):
        off = tl.load(pts + tl.arange(0, W))
        return tl.sum(tl.load(base + e * stride + off, cache_modifier=".cg").to(tl.int32), axis=0)

    @triton.jit
    def _touch_pages(ids, b0, b1, b2, b3, s0, s1, s2, s3, pts, sink, W: tl.constexpr):
        i = tl.program_id(0)
        e = tl.load(ids + i).to(tl.int64)
        acc = _touch_region(b0, s0, pts, e, W) + _touch_region(b1, s1, pts + W, e, W)
        acc += _touch_region(b2, s2, pts + 2 * W, e, W) + _touch_region(b3, s3, pts + 3 * W, e, W)
        tl.store(sink + i, acc)  # keeps the loads alive


class Toucher:
    """Touches page points of four byte regions, each indexed by ids[i] * stride (stride 0 = whole region)."""

    def __init__(self, regions, sizes, device):
        if len(regions) != 4:
            raise ValueError("Toucher takes four (bytes, stride) regions")
        self.regions = regions
        table, self.width = points_table(sizes)
        self.pts = table.to(device)
        self.sink = torch.zeros(4096, dtype=torch.int32, device=device)

    def __call__(self, ids: torch.Tensor):
        (b0, s0), (b1, s1), (b2, s2), (b3, s3) = self.regions
        _touch_pages[(ids.numel(),)](ids, b0, b1, b2, b3, s0, s1, s2, s3, self.pts, self.sink, W=self.width)


# ---------------------------------------------------------------- rings

def moe_route_sets(device, m, d, *, banks, window):
    """Call j uses bank j % banks; its M tokens route to exactly d distinct experts, a window that advances
    by d per call through the first `window` experts of the bank. Returns [(bank, topk_ids)].
    One bank (the small ring): the window is rounded up to a multiple of d so consecutive calls never share
    experts (a shared expert could still sit in L2 and flatter the small ring)."""
    if banks == 1:
        window = d * -(-window // d)
    if d > window or window > E or d < TOPK or d > m * TOPK:
        raise ValueError(f"cannot route M={m} tokens to exactly {d} distinct experts in a {window}-expert window")
    per_bank = window // d if banks == 1 else -(-window // d) + 1
    routes = torch.arange(m * TOPK).reshape(m, TOPK) % d
    out = []
    for i in range(per_bank):
        for b in range(banks):
            ids = ((routes + i * d) % window).to(torch.int32).to(device).contiguous()
            out.append((b, ids))
    return out


def ring_footprint(calls, bytes_per_item) -> int:
    """Distinct bytes a ring touches: calls is [(bank, ids)], one item per distinct (bank, id)."""
    return len({(b, int(e)) for b, ids in calls for e in ids.flatten().tolist()}) * bytes_per_item


# ---------------------------------------------------------------- timing

def time_calls(calls, *, side=None, repeat=1, replays=20, warmup=3):
    """Median us per call of one graph that runs calls[i]() `repeat` times each; side[i]() (if given) runs on a
    low-priority stream forked before call i and joined after it."""
    main = torch.cuda.Stream(priority=-1)
    side_stream = torch.cuda.Stream(priority=0)

    def body():
        cur = torch.cuda.current_stream()
        for i, run in enumerate(calls):
            if side is not None:
                side_stream.wait_stream(cur)
                with torch.cuda.stream(side_stream):
                    side[i]()
            for _ in range(repeat):
                run()
            if side is not None:
                cur.wait_stream(side_stream)

    with torch.cuda.stream(main):
        body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=main):
        body()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(replays):
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1e3 / len(calls))
    graph.reset()
    return statistics.median(samples)


def run_configs(calls_small, calls_big, side, replays):
    """calls_*: list of run callables; side(ring, mode, i) -> touch callable. Returns {config: us/call}."""
    res = {}
    for ring, calls in (("r512", calls_small), ("r8g", calls_big)):
        n = len(calls)
        single = time_calls(calls, replays=replays)
        res[ring] = single
        res[ring + "-warm"] = time_calls(calls, repeat=2, replays=replays) - single  # pair - single
        for mode in ("same", "next"):
            if side(ring, mode, 0) is not None:
                res[f"{ring}-{mode}"] = time_calls(calls, side=[side(ring, mode, i) for i in range(n)],
                                                   replays=replays)
    res["r512 again"] = time_calls(calls_small, replays=replays)
    return res


# ---------------------------------------------------------------- MoE

def _moe_bank(device, seed):
    from b12x._lib.intrinsics import swizzle_block_scale
    from b12x.moe import fused_moe

    gen = torch.Generator(device=device).manual_seed(seed)
    w1 = torch.randint(0, 256, (E, 2 * I, K // 2), dtype=torch.uint8, device=device, generator=gen)
    w2 = torch.randint(0, 256, (E, K, I // 2), dtype=torch.uint8, device=device, generator=gen)

    def scales(shape):
        exp = torch.randint(-9, -4, shape, device=device, generator=gen).float()
        return swizzle_block_scale(torch.exp2(exp).to(torch.float8_e4m3fn)).contiguous()

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
        input_scale=torch.full((1,), 48.0, device=device),
        intermediate_scale=torch.full((1,), 384.0, device=device),
    ))


def _moe_toucher(experts, device):
    impl = experts._impl
    regions, sizes = [], []
    for name in ("w1_fp4", "w1_blockscale", "w2_fp4", "w2_blockscale"):
        t = getattr(impl, name)
        raw = as_bytes(t)
        if t.shape[0] != E or raw.numel() % E:
            print(f"warning: {name} {tuple(t.shape)} is not expert-major; touch addresses are approximate")
        regions.append((raw, raw.numel() // E))
        sizes.append(raw.numel() // E)
    return Toucher(regions, sizes, device)


def bench_moe(device, shapes, banks, small_bytes, replays):
    from b12x.moe import fused_moe
    from b12x.preparation import PreparationSession, PreparedCall

    bank_list = [_moe_bank(device, seed=b + 1) for b in range(banks)]
    touchers = [_moe_toucher(x, device) for x in bank_list]
    window_small = -(-small_bytes // EXPERT_BYTES)
    print(f"MoE: {banks} banks x {E} experts x {EXPERT_BYTES} B = {banks * E * EXPERT_BYTES / 2**30:.2f} GiB; "
          f"small ring = {window_small} experts of bank 0", flush=True)
    results = {}
    for m, d in shapes:
        a = (torch.randn(m, K, device=device) * 0.35).to(torch.bfloat16)
        weights = torch.full((m, TOPK), 1.0 / TOPK, device=device)
        small = moe_route_sets(device, m, d, banks=1, window=window_small)
        big = moe_route_sets(device, m, d, banks=banks, window=E)
        override = fused_moe.MoeDecodeConfig(
            backend="dynamic", route_planner="internal", max_active_clusters=None,
            dynamic_tile_m=16, dynamic_route_mode="grouped", nvfp4_share_input=True,
        )
        plans = [fused_moe.plan_execution(
            experts=x, capacity=fused_moe.ExecutionCapacity(max_tokens=m, top_k=TOPK),
            invocation={"fast_math": True}, override=override) for x in bank_list]
        out = torch.empty(m, K, dtype=torch.bfloat16, device=device)

        def factory(state, ids=big[0][1]):
            sc = tuple(torch.empty(s.shape, dtype=s.dtype, device=device) for s in state.scratch.scratch_specs())
            o = torch.empty_like(a)
            b = state.bind(a=a, topk_ids=ids, topk_weights=weights, scratch=sc, output=o, input_scales_static=True)
            return PreparedCall(run=b.run, output=o, owners=(b, sc))

        with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
            session.prepare(tuple(p.request(name=f"pagewalk-moe-b{b}-m{m}", prepare_call=factory)
                                  for b, p in enumerate(plans)))
            session.freeze()
            scratch = [tuple(torch.empty(s.shape, dtype=s.dtype, device=device) for s in p.scratch_specs())
                       for p in plans]

            def runs(route):
                out_runs = []
                for b, ids in route:
                    bind = fused_moe.bind(plans[b], a=a, topk_ids=ids, topk_weights=weights, scratch=scratch[b],
                                          output=out, input_scales_static=True)
                    out_runs.append(lambda bind=bind: fused_moe.run(binding=bind))
                return out_runs

            routes = {"r512": small, "r8g": big}

            def side(ring, mode, i):
                route = routes[ring]
                b, ids = route[i if mode == "same" else (i + 1) % len(route)]
                return lambda: touchers[b](ids)

            small_runs = runs(small)
            small_runs[0]()
            check_output(out, f"moe M={m}")
            res = run_configs(small_runs, runs(big), side, replays)
        foot = {"r512": ring_footprint(small, EXPERT_BYTES), "r8g": ring_footprint(big, EXPERT_BYTES)}
        _report("moe", m, d, res, d * EXPERT_BYTES, foot)
        results[m] = res
    return results


# ---------------------------------------------------------------- dense MXFP8

def bench_dense(device, ms, copies_big, small_bytes, replays):
    from b12x.gemm import blockscaled
    from b12x.gemm.blockscaled import _a16
    from b12x.preparation import PreparationSession, PreparedCall

    n, k = DENSE_N, K
    gen = torch.Generator(device=device).manual_seed(7)
    weights = []
    for _ in range(copies_big):
        values = (torch.randn(n, k, device=device, generator=gen) * 2).to(torch.float8_e4m3fn)
        exponent = torch.randint(120, 133, (n, k // 32), device=device, dtype=torch.uint8, generator=gen)
        weights.append(blockscaled.pack_weight(values, exponent))
    parts = [_a16._weight_parts(w)[:2] for w in weights]
    wbytes = sum(t.numel() * t.element_size() for t in parts[0])
    copies_small = -(-small_bytes // wbytes)
    print(f"dense MXFP8 {n}x{k}: {wbytes / 1e6:.1f} MB per copy; rings {copies_small} / {copies_big} copies "
          f"({copies_small * wbytes / 2**20:.0f} MiB / {copies_big * wbytes / 2**30:.2f} GiB)", flush=True)
    sizes = [t.numel() * t.element_size() for t in parts[0]] * 2
    touchers = [Toucher([(as_bytes(v), 0), (as_bytes(s), 0)] * 2, sizes, device) for v, s in parts]
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    results = {}
    for m in ms:
        source = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.25
        query = blockscaled.query_from_call(source, weights[0], activation_mode="auto", expected_m=m)
        plan = blockscaled.plan(query)
        values, scales, gscale, _ = _a16._weight_parts(weights[0])
        request = plan.request(name=f"pagewalk-dense-m{m}", prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(source, values, scales, gscale, activation_scale=None, out=None, workspace=None)))
        with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
            session.prepare((request,))
            runs = [lambda w=w: blockscaled.mm(source, w, plan=plan) for w in weights]
            small, big = runs[:copies_small], runs
            check_output(small[0](), f"dense M={m}")

            def side(ring, mode, i):
                if mode != "next":
                    return None
                count = copies_small if ring == "r512" else copies_big
                return lambda j=(i + 1) % count: touchers[j](zero)

            res = run_configs(small, big, side, replays)
        _report("dense", m, None, res, wbytes, {"r512": copies_small * wbytes, "r8g": copies_big * wbytes})
        results[m] = res
    return results


# ---------------------------------------------------------------- report

def check_output(out, label):
    """Correctness gate before timing: finite and nonzero output from the real kernel path."""
    torch.cuda.synchronize()
    if not torch.isfinite(out).all() or not torch.count_nonzero(out):
        raise RuntimeError(f"{label}: output is nonfinite or all zero")
    print(f"{label}: output finite, {torch.count_nonzero(out).item()}/{out.numel()} nonzero", flush=True)


def _report(kind, m, d, res, bytes_per_call, foot):
    print(f"\n{kind} M={m}" + (f" D={d}" if d else "") + f", {bytes_per_call / 1e6:.1f} MB/call, "
          f"footprint r512 {foot['r512'] / 2**20:.0f} MiB, r8g {foot['r8g'] / 2**30:.2f} GiB")
    print(f"  {'config':12} {'us/call':>9} {'GB/s':>7} {'vs r512':>8} {'vs r8g':>8}")
    for name, us in res.items():
        print(f"  {name:12} {us:9.1f} {bytes_per_call / (us * 1e3):7.1f} "
              f"{100 * (us / res['r512'] - 1):+7.1f}% {100 * (us / res['r8g'] - 1):+7.1f}%", flush=True)


def summarize(res):
    """TLB term and touch gain of one (kind, M) result, in us/call and fractions."""
    # r512 runs before and after the r8g configs: their mean cancels linear clock drift
    base = (res["r512"] + res.get("r512 again", res["r512"])) / 2
    mode = min(("same", "next"), key=lambda k: res.get(f"r8g-{k}", math.inf))
    gap = res["r8g"] - base
    gain = res["r8g"] - res.get(f"r8g-{mode}", math.inf)
    return dict(
        gap_us=gap, gap_pct=100 * gap / base, gain_us=gain, gain_frac=gain / res["r8g"],
        cost_us=res.get(f"r512-{mode}", base) - base, mode=mode,
        recovered=gain / gap if gap > 0 else 0.0,
        go=gap >= 0.05 * base and gain >= 0.5 * gap,
    )


def projection(moe, dense, dense_m):
    print("\nexpected ms/step at TP=1 (k18 profile after keepalive; 48 MoE calls/step; dense GEMVs scaled from "
          f"the {DENSE_N}x{K} shape at the same M)")
    print("  'like r8g' = serving behaves like the 8 GiB ring (dgpp's in-step finding);"
          " 'like r512' = no TLB term, the touch only adds its cost")
    for m, (cell, moe_ms, dense_ms, step_ms) in PROFILE.items():
        parts, cost, go = [], 0.0, []
        saved = 0.0
        if m in moe:
            s = summarize(moe[m])
            saved += MOE_CALLS_PER_STEP * s["gain_us"] / 1e3
            cost += MOE_CALLS_PER_STEP * s["cost_us"] / 1e3
            parts.append(f"MoE TLB term {s['gap_pct']:+.1f}% ({MOE_CALLS_PER_STEP * s['gap_us'] / 1e3:+.2f} ms), "
                         f"touch -{MOE_CALLS_PER_STEP * s['gain_us'] / 1e3:.2f} ms ({100 * s['recovered']:.0f}% of term)")
            go.append(f"moe {'GO' if s['go'] else 'no-go'}")
        dm = dense_m.get(m)
        if dm in dense:
            s = summarize(dense[dm])
            saved += dense_ms * s["gain_frac"]
            cost += dense_ms * s["cost_us"] / (dense[dm]["r8g"] - s["gap_us"])
            parts.append(f"dense TLB term {s['gap_pct']:+.1f}%, touch -{dense_ms * s['gain_frac']:.2f} ms")
            go.append(f"dense {'GO' if s['go'] else 'no-go'}")
        if not parts:
            continue
        print(f"  {cell} (M={m}, step {step_ms} ms, MoE {moe_ms}, dense {dense_ms}): " + "; ".join(parts))
        print(f"      like r8g: {-saved:+.2f} ms/step ({-100 * saved / step_ms:+.1f}%); "
              f"like r512: {cost:+.2f} ms/step; " + ", ".join(go))


# ---------------------------------------------------------------- CPU self-test

def selftest():
    import random
    rng = random.Random(0)
    for _ in range(2000):
        size = rng.choice([1, 2, PAGE - 1, PAGE, PAGE + 1, 3 * PAGE, rng.randrange(1, 50 * PAGE)])
        base = rng.randrange(0, 64 * PAGE)
        pages = set(range(base // PAGE, (base + size - 1) // PAGE + 1))
        pts = page_points(size)
        assert all(0 <= p < size for p in pts)
        assert {(base + p) // PAGE for p in pts} == pages, (size, base)
    table, w = points_table([1638400, 204800, 819200, 102400])
    assert w == 2 and table.tolist()[0] == [0, 1638399] and table.tolist()[3] == [0, 102399]
    table, w = points_table([DENSE_N * K, DENSE_N * K // 32] * 2)
    assert w == 32 and table[0, 19].item() == 19 * PAGE and table[0, 20].item() == DENSE_N * K - 1  # 20 pages exactly
    assert table[0, 31].item() == DENSE_N * K - 1
    assert EXPERT_BYTES == 2_764_800
    small_window = -(-(512 << 20) // EXPERT_BYTES)
    assert small_window == 195
    for m, d in ((5, 33), (20, 96), (40, 151)):
        small = moe_route_sets("cpu", m, d, banks=1, window=small_window)
        big = moe_route_sets("cpu", m, d, banks=7, window=E)
        for _, ids in small + big:
            assert ids.shape == (m, TOPK) and ids.unique().numel() == d and int(ids.max()) < E
        assert ring_footprint(small, EXPERT_BYTES) == d * -(-small_window // d) * EXPERT_BYTES >= 512 << 20
        for (_, x), (_, y) in zip(small, small[1:] + small[:1], strict=True):
            assert not set(x.flatten().tolist()) & set(y.flatten().tolist())  # neighbours disjoint
        assert ring_footprint(big, EXPERT_BYTES) == 7 * E * EXPERT_BYTES >= 8 << 30
        assert len({b for b, _ in big[:7]}) == 7  # consecutive calls hit different banks
    fake = {"r512": 100.0, "r8g": 110.0, "r8g-same": 104.0, "r8g-next": 103.0, "r512-next": 101.0,
            "r512-same": 105.0, "r512 again": 100.0}
    s = summarize(fake)
    assert abs(s["gap_pct"] - 10) < 1e-9 and abs(s["recovered"] - 0.7) < 1e-9 and s["go"]
    assert s["mode"] == "next" and abs(s["cost_us"] - 1.0) < 1e-9
    assert abs(summarize({**fake, "r512 again": 102.0})["gap_us"] - 9.0) < 1e-9  # drift-averaged base
    assert not summarize({**fake, "r8g": 104.0, "r8g-same": 103.0, "r8g-next": 103.5})["go"]  # 4% gap
    projection({20: fake}, {20: fake}, {20: 20})
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert as_bytes(x).numel() == 48 and as_bytes(x.T).numel() == 48
    assert as_bytes(x.T).data_ptr() == x.data_ptr()
    print("selftest ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--part", default="moe,dense")
    parser.add_argument("--shapes", default="5:33,20:96,40:151", help="MoE M:D (tokens:distinct experts)")
    parser.add_argument("--ring-gib", type=float, default=8.0)
    parser.add_argument("--small-mib", type=int, default=512)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    device = torch.device("cuda")
    try:
        from benchmarks.common import nvidia_smi_gpu_mode_snapshot
        print("gpu mode:", nvidia_smi_gpu_mode_snapshot(), flush=True)
    except Exception as error:  # informational only
        print("gpu mode: unavailable:", error)
    os.environ.pop("B12X_MOE_DECODE_BACKEND", None)
    shapes = [tuple(int(v) for v in s.split(":")) for s in args.shapes.split(",")]
    ring, small = int(args.ring_gib * 2**30), args.small_mib << 20
    parts = args.part.split(",")
    moe = bench_moe(device, shapes, -(-ring // (E * EXPERT_BYTES)), small, args.replays) if "moe" in parts else {}
    torch.cuda.empty_cache()
    dense = {}
    if "dense" in parts:
        dense_bytes = DENSE_N * K * 33 // 32
        dense = bench_dense(device, [m for m, _ in shapes], -(-ring // dense_bytes), small, args.replays)
    projection(moe, dense, {m: m for m, _ in shapes})


if __name__ == "__main__":
    main()
