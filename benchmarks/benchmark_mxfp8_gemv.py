#!/usr/bin/env python3
"""MXFP8 decode GEMV (mode "gemv") against the raced A16/quantized plans.

For each Qwen3.8 Flash Next dense shape and live row count, every A16 tile/split
configuration and the quantized path are prepared (the autotuner's race), and
every GEMV split/CTA configuration. Each candidate is captured in one CUDA graph
that cycles through enough weight copies to exceed L2, so a call reads its
weights from DRAM the way back-to-back serving layers do. Reported: median us
per call over repeated paired replays, achieved GB/s over weight+scale bytes,
and the GEMV best / current best ratio. Outputs are checked against an FP32
oracle before timing.

  B12X_DENSE_GEMV=1 python3 benchmarks/benchmark_mxfp8_gemv.py --tp 1 --rows 1 5 8
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from b12x.gemm import blockscaled
from b12x.gemm.blockscaled import _a16, _gemv, _preparation, _tuning
from b12x.gemm.blockscaled._tuning import BlockscaledConfig
from b12x.preparation import PreparationSession, PreparedCall
from b12x._lib.runtime_control import kernel_resolution_guard

# (name, N, K) per rank. HC mixers are replicated, so TP=1 and TP=2 share them.
HC = [("hc.mix_down", 336, 10240), ("hc.mix_up", 10240, 320), ("hc.final_down", 320, 10240)]
SHAPES = {
    1: [("gdn.in_proj_qkvz", 16384, 2560), ("attn.qkv", 13312, 2560), ("gdn.out/attn.o", 2560, 6144),
        ("gdn.in_proj_ba", 96, 2560), ("shared.gate_up", 1280, 2560), ("shared.down", 2560, 640),
        ("indexer.qk", 640, 2560), *HC],
    2: [("gdn.in_proj_qkvz", 8192, 2560), ("attn.qkv", 6656, 2560), ("gdn.out/attn.o", 2560, 3072),
        ("gdn.in_proj_ba", 48, 2560), ("shared.gate_up", 640, 2560), ("shared.down", 2560, 320),
        ("indexer.qk", 640, 2560), *HC],
}
A16 = tuple(itertools.product((64, 128), (64, 128), (1, 2, 4, 8)))
L2_TARGET = 96 << 20


def weight(n, k):
    values = (torch.randn(n, k, device="cuda") * 2).to(torch.float8_e4m3fn)
    exponent = torch.randint(118, 124, (n, k // 32), device="cuda", dtype=torch.uint8)
    packed = blockscaled.pack_weight(values, exponent)
    decoded = values.float() * torch.exp2(exponent.float() - 127).repeat_interleave(32, 1)
    return packed, decoded


def candidates(query, device):
    out = [("quantized", BlockscaledConfig(mode="quantized"))]
    out += [(f"a16_{n}_{k}_{s}", BlockscaledConfig(mode="a16", tile_n=n, tile_k=k, split_k=s)) for n, k, s in A16]
    if _tuning.gemv_eligible(query, device):
        out += [(f"gemv_s{s}_c{c}", BlockscaledConfig(mode="gemv", gemv_split=s, gemv_ctas=c))
                for s in _gemv.SPLITS for c in _gemv.CTAS_PER_SM]
    legal, seen = [], set()
    for label, config in out:
        try:
            _tuning._validate_config(query, config, device)
        except ValueError:
            continue
        key = repr(_tuning._equivalence(query, device, config))
        if key not in seen:
            seen.add(key)
            legal.append((label, config))
    return legal


def bench_shape(name, n, k, m, args, device):
    torch.manual_seed(n * 7 + k + m)
    packed, decoded = weight(n, k)
    nbytes = n * k + n * (k // 32)
    copies = max(1, min(args.max_copies, -(-L2_TARGET // nbytes)))
    weights = [packed] + [weight(n, k)[0] for _ in range(copies - 1)]
    source = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    reference = source.float() @ decoded.T
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    query = blockscaled.query_from_call(source, packed, activation_mode="auto", out=out, expected_m=m)
    cands = candidates(query, device)
    need = max(_preparation._workspace_bytes(query, c) for _, c in cands)
    scratch = torch.empty(max(need, 16), device="cuda", dtype=torch.uint8)
    query = blockscaled.query_from_call(source, packed, activation_mode="auto", out=out,
                                        workspace=scratch, expected_m=m)
    plans, requests = {}, []
    for label, config in cands:
        for i, w in enumerate(weights):
            plan = blockscaled.plan(query, override=config)
            values, scales, gs, _ = _a16._weight_parts(w)
            requests.append(plan.request(name=f"{name}.{label}.{i}", prepare_call=lambda state, v=values, s=scales: PreparedCall(
                run=lambda: state.run(source, v, s, None, out=out, workspace=scratch))))
            plans[(label, i)] = plan
    with PreparationSession(device=source.device, autotune=False, compile_workers=args.workers) as session:
        session.prepare(tuple(requests))
        graphs, errors = {}, {}
        with kernel_resolution_guard("gemv bench"):
            for label, _ in cands:
                out.fill_(float("nan"))
                blockscaled.mm(source, packed, out=out, workspace=scratch, plan=plans[(label, 0)])
                rel = float(torch.linalg.vector_norm(out.float() - reference) / torch.linalg.vector_norm(reference))
                errors[label] = rel
                if not rel < (5e-2 if label == "quantized" else 5e-3):  # MXFP8 activations
                    raise RuntimeError(f"{name} M={m} {label}: relative error {rel}")
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for r in range(args.reps):
                        for i, w in enumerate(weights):
                            blockscaled.mm(source, w, out=out, workspace=scratch, plan=plans[(label, i)])
                graphs[label] = graph
            calls = args.reps * len(weights)
            for g in graphs.values():
                g.replay()
            torch.cuda.synchronize()
            samples = {label: [] for label in graphs}
            for t in range(args.iters):
                order = list(graphs) if t % 2 == 0 else list(graphs)[::-1]
                for label in order:
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record(); graphs[label].replay(); b.record()
                    b.synchronize()
                    samples[label].append(a.elapsed_time(b) * 1000 / calls)
    med = {label: statistics.median(v) for label, v in samples.items()}
    current = min((l for l in med if not l.startswith("gemv")), key=med.get)
    gemvs = [l for l in med if l.startswith("gemv")]
    row = dict(name=name, n=n, k=k, m=m, mb=nbytes / 1e6, copies=copies,
               current=current, current_us=med[current], current_gbs=nbytes / med[current] / 1e3)
    if gemvs:
        best = min(gemvs, key=med.get)
        row.update(gemv=best, gemv_us=med[best], gemv_gbs=nbytes / med[best] / 1e3,
                   ratio=med[best] / med[current], gemv_rel_err=errors[best],
                   gemv_default=f"gemv_s{_gemv.default_split(n, k, m, device.sm_count)}_c2",
                   gemv_default_us=med.get(f"gemv_s{_gemv.default_split(n, k, m, device.sm_count)}_c2"))
    row["all_us"] = med
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, nargs="+", default=[1, 2])
    p.add_argument("--rows", type=int, nargs="+", default=[1, 5, 8])
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--max-copies", type=int, default=24)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", type=pathlib.Path, required=True)
    args = p.parse_args()
    if not _tuning._DENSE_GEMV:
        raise SystemExit("set B12X_DENSE_GEMV=1")
    from b12x.preparation.device import detect_device
    device = detect_device(torch.device("cuda")).identity
    with args.out.open("a") as fh:
        for tp in args.tp:
            for name, n, k in SHAPES[tp]:
                if args.only and name not in args.only:
                    continue
                for m in args.rows:
                    row = bench_shape(name, n, k, m, args, device)
                    row["tp"] = tp
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    print(f"TP{tp} {name:18s} N={n:6d} K={k:5d} M={m:2d} current {row['current']:16s} "
                          f"{row['current_us']:8.2f}us {row['current_gbs']:6.1f}GB/s"
                          + (f" | gemv {row['gemv']:12s} {row['gemv_us']:8.2f}us {row['gemv_gbs']:6.1f}GB/s"
                             f" ratio {row['ratio']:.3f}" if "gemv" in row else " | gemv n/a"), flush=True)
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
