# flint: byte-balanced NVFP4 MoE decode (TP=1, I=640)

Branch `exp/flint` (base `5bf24021` = `b7fbaf96` + the `wm` backend, the b12x in the shipped
TP=1 image). Target: Qwen3.8-Flash-Next routed experts on one GB10 (48 SMs, LPDDR5X),
512 experts, top-10, hidden K=2560, intermediate I=640, decode batches M=5..40
(c1..c8 verify steps with 4 MTP drafts).

## Phase 0: accounting

Bytes per touched expert at I=640:

| format | weights | scales | total |
|---|---:|---:|---:|
| NVFP4 (E4M3 per 16) | 2,457,600 | 307,200 (11.1%) | 2,764,800 |
| INT4 g128 (FP16 per 128, reference stack) | 2,457,600 | 76,800 | 2,534,400 (-8.3%) |

Distinct experts D per M (`results/r4-dprobe-20260927`, pooled fresh/16k/count workloads):
M5 33.0, M10 57.6, M15 79.4, M20 93.7 (sd 22.8), M40 150.6.

Per layer, ideal time at 244 GB/s against the b12x `dynamic` kernel in the microbench
(`results/r8-wm-bench`, balanced synthetic routing):

| M | D | MB | ideal us | dynamic us | % of 244 GB/s |
|---:|---:|---:|---:|---:|---:|
| 5 | 33 | 91.2 | 374 | 444-469 | 80-84 |
| 10 | 60 | 165.9 | 680 | 802-809 | 84-85 |
| 20 | 96 | 265.4 | 1,088 | 1,194-1,262 | 86-91 |
| 40 | 151 | 417.5 | 1,711 | not measured | |

Practical ceiling. The r5 stream probe (`benchmarks/benchmark_stream_ceiling.py`) reads
239-243 GB/s for one sequential stream. With a second scale-plane stream it prints
188-218, but that column counts payload bytes only. With the scale bytes included,
every scale-copy layout (4 B gathers, 8 B pairs, linear 16 B) moves 235-240 GB/s
in total. The separate scale plane therefore does not slow the copy beyond its own
bytes, and the ceiling to plan against is about 240 GB/s.

Serving. In the TP=1 profile (`results/r8-prof`, fresh c4, M=20) the MoE kernel takes
1,462 us per layer (median 1,455; p10 1,195; p90 1,738). The bench takes 1,194-1,262 us at
D=96. The reference stack (Marlin W4A16, `results/uf-prof-20261002`, fresh c4, M=16 with
3 drafts) takes 895 us per layer for its FC1 and FC2 launches together. Both terms
behind the bytes gap work against us:

- Format: 2.76 vs 2.53 MB per expert (+9.1%).
- Drafts: 4 drafts at c4 give 20 tokens per layer, 3 drafts give 16. The interpolated D
  ratio is 0.84-0.88, so about 13% fewer experts.

The remaining difference is kernel efficiency under real routing, against about 95% of
roofline for Marlin.

Gate. The roofline gap at M=10-40 is 9-15% of MoE time. Packed scales would cut another
4.2% of bytes. The total is >= 8%, so the work went ahead. Packed scales are deferred
(see Not done).

## Why dynamic and wm leave time on the table

`dynamic` (one persistent launch) runs a queue of (expert, m16 tile, 128-channel slice)
tasks: 5 slices per expert at I=640 on 48 CTAs. Its efficiency follows the round
quantization of tasks / 48:

| M | D | tasks | rounds | dynamic % of 240 GB/s |
|---:|---:|---:|---:|---:|
| 5 | 33 | 165 | 3.44 | 82-85 |
| 10 | 60 | 300 | 6.25 | 86 |
| 20 | 96 | 480 | 10.00 | 92 |

M20/D96 is a whole number of rounds, which is the best case. In serving D changes from
call to call (p10-p90 above), so the round tail costs every layer some time on average.

`wm` (`docs/moe-weight-major-decode.md`) gives each CTA whole experts. It has the same
problem, in coarser form: ceil(D / 48) rounds, and at D=60 only 12 CTAs stream the
second round. Its timing probes support this. Without compute it runs at the same speed.
Without scales it gains exactly the scale bytes. So it is limited by the schedule, not
by the copy or the math.

## Design

flint keeps `wm`'s stage stream, QMMA path and numerics and changes only the work
split. Every call is cut into equal byte ranges over all CTAs, whatever D is:

1. **Routing.** Every CTA builds the touched-expert bitmaps from `topk_ids` in shared
   memory. This is `wm`'s prologue, now `_route_prologue`. An item is (expert, 16-row
   pass), as in `wm`.
2. **Phase A (FC1).** The unit is (item, 8-channel block): 8 up rows and the same 8 gate
   rows over full K. These are two `wm` FC1 stages of 10 KB contiguous payload plus
   1,280 B of scales each. The `items x 80` units are cut into 48 contiguous ranges that
   differ by at most one unit (22.5 KB, about 0.4% of a CTA's share at D=96). Each
   unit ends with SiLU(alpha g) x (alpha u), rounded to BF16 and stored to a global h
   row per route (`hbuf[route][I]`).
3. **Grid barrier.** It is sense-reversing, uses slot 0 of the workspace's
   `barrier_count` / `barrier_epoch`, and leaves the counter at zero. All CTAs are
   resident: 1 CTA per SM, grid <= SMs. Weight addresses depend only on routing, so the
   cp.async ring has already issued the first FC2 stages before a CTA waits.
4. **Phase B (FC2).** The unit is (item, 32 down rows) over full I, one `wm` FC2 stage of
   10 KB contiguous plus scales. The `items x 80` units are cut the same way. When the
   item changes, the CTA loads that item's h rows from L2 (`ld.global.cg`), requantizes
   them with `wm`'s per-16 quantizer, runs `wm`'s FC2 stage and then the BF16x2 atomic
   scatter-add. That is one atomic add per (routed expert, output element), the same
   as `wm`.

Fused FC1 -> SiLU -> FC2 per expert, with the intermediate kept on chip, was `wm`'s
design. The per-expert fusion is what ties a CTA to a whole expert and creates the
round tail, so flint gives it up. The intermediate is M x 10 x 640 BF16 (256 KB at M=20)
and stays in L2.

### Numerics

flint matches `wm` except for the order of the output atomics. Both kernels use the same
activation quantizer and input scale, the same `m16n8k64 kind::mxf4nvf4` QMMA and K
order, the same 4-warp split-K reduction order, the same SiLU and BF16 rounding of h,
and the same FC2 requantizer and epilogue. `wm` itself matches `dynamic` within its
combine noise (`tests/moe/test_wm_decode.py`).

### Correctness

- `tests/moe/test_wm_decode.py` with `B12X_MOE_WM_SCHEDULE=flint` checks flint against
  `dynamic` and against the FP32-accumulating NVFP4 oracle (`moe_reference_nvfp4`). It
  covers capacities 5/20/32, the spread/clustered/hot (two-pass)/disjoint routing
  patterns, w31 storage, and graph replay with live route changes and no replay
  allocation, at I=640 and I=320.
- `tests/moe/test_flint_geometry.py` (CPU) checks shared memory, scratch fit, that the
  ranges tile the work with at most one unit of imbalance, and the selection controls.

### Plan and cache-key integration

- Selection: `B12X_MOE_DECODE_BACKEND=wm` plus `B12X_MOE_WM_SCHEDULE=flint`. The
  schedule joins the wm query controls (`wm_schedule`) only when set, so the default
  and plain-wm keys stay byte-identical. The kernel cache key is
  `("flint_decode", 1, ...)`, compiled under `integration.tp_moe.flint_decode` and
  precompiled by the same preparation path as `wm`.
- Scratch: no new allocation. h uses the shadow dynamic workspace's `packed_input`
  (rows_padded x K/2 bytes, which holds routes x I x 2 whenever 2I <= K/2). The barrier
  uses `barrier_count` / `barrier_epoch`.
- vLLM: declare `B12X_MOE_WM_SCHEDULE` (and `B12X_MOE_DECODE_BACKEND`) in `vllm/envs.py`
  so the torch AOT compile key changes with them (see the plan-population gotcha).
- Known limit: the launcher reads the schedule from the environment at launch, as
  `B12X_MOE_WM_GRID` does. One process must not mix wm and flint plans. Moving the
  schedule into the plan's decode config needs a config-schema bump and is left for
  promotion.

## Expected result

At about 235 GB/s effective plus ~10 us of fixed cost (routing, one barrier, output
memset):

| M | D | dynamic us (bench) | flint us (model) | change |
|---:|---:|---:|---:|---:|
| 5 | 33 | 444-469 | ~400 | -10..-15% |
| 10 | 60 | 802 | ~715 | -11% |
| 20 | 96 | 1,194-1,257 | ~1,140 | -5..-9% |
| 32 | 133 | ~1,700 | ~1,575 | ~-7% |

In serving, D varies per call, so dynamic pays its round tail on average and flint does
not. The serving gain should therefore be at least the bench gain at the same mean D.
At c4 (48 layers) -100 us/layer is -4.8 ms/step.

## Not done (and why)

- **M > 32** (c8 at 4 drafts is M=40): `wm` geometry allows two 16-row passes. Supporting
  it needs a third pass, or 32-row passes (QMMA M16 x 2).
- **Packed scales** (base byte + 4-bit delta per 8-block group): 99.4% of groups fit
  losslessly (measured on layers 3/24/45, entropy 3.4 bits) and the change would save 4.2%
  of bytes. A second copy costs 7.5 GB at TP=1, which is not available. The prefill
  kernels would also have to read the packed planes, and the 0.6% of escape groups
  change numerics, so this needs its own gate.
- **FP4 MMA vs A16**: flint keeps `wm`'s native FP4 QMMA. `wm:nocompute` shows that compute
  is not on the critical path at M <= 32. The A16 route is opus-kernel-18's microbench.
- **TMA bulk copies**: the cp.async ring already reaches the stream ceiling in the probe.
  Change it only if flint's measured rate stays well below about 235 GB/s.

## Microbench

`benchmarks/benchmark_moe_wm.py --intermediate 640 --backends dynamic,wm,flint`. The
default is balanced synthetic routing. `--skew S` uses Zipf(S)-skewed routing, with D in
`--shapes` taken as the window size and the realized mean D reported. Results: see the
journal (opus-kernel-19).
