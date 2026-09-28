# Weight-major NVFP4 MoE decode (`wm` backend)

Branch `feat/r5-fused-moe-decode` from `b7fbaf96`. Target: Qwen3.8-Flash-Next NVFP4
routed experts on GB10 (SM121, 48 SMs, LPDDR5X 273 GB/s spec, ~249 GB/s reached by
the lm_head GEMVs), TP=2.

## Problem

Per node and layer: E=512 experts, hidden K=2560, intermediate I=320 (640/TP),
top-k 10. One touched expert costs 1,382,400 B per node (w13 819,200 + its E4M3
scales 102,400; down 409,600 + scales 51,200). Measured on b1.2 (profile
`results/r4-dbg-20260927-2102/prof`, expert probe `mods/vllm-moe-dprobe`):

| | M rows | distinct experts D | bytes/layer | MoE ms/step | per layer | achieved |
|---|---:|---:|---:|---:|---:|---:|
| c1 | 5 | 33.0 | 45.6 MB | 13.1 | 273 us | ~172 GB/s |
| c4 | 20 | 95.6 | 132.2 MB | 35.0 | 729 us | ~182 GB/s |

At 249 GB/s the floor is 183 us (c1) and 531 us (c4) per layer.

## Why the current kernel leaves bandwidth on the table

The `dynamic` kernel (tile M16, grouped, internal planner) runs one persistent launch:
phase 0/1 route + pack every routed row into an expert-major packed-A domain, a
resident-grid barrier, then a queue of `(expert, m_tile, 128-wide slice)` tasks. With
I=320 each expert splits into slices of 128/128/64 channels, so c1 has 99 tasks of two
sizes on 48 CTAs, each task pays a fresh pipeline fill at FC1 start and at the
FC1->FC2 turn, and nothing streams during the route/pack phase and barrier. Each
output element takes 30 bf16 atomic adds (10 experts x 3 slices).

## Design

One launch, `grid = SM count` (48), 1 CTA/SM, 4 warps, **no inter-CTA communication
except the output atomics** (no barrier, no queue, no counters to reset).

1. **Routing in every CTA, from smem.** Each CTA reads `topk_ids` (<= 32 x 10), builds a
   512-bit touched-expert bitmap, ranks experts in ascending id order, and gathers each
   rank's rows (token, router weight). Work item = (rank, pass), pass = 16-row block of
   that expert's rows (a second pass only when > 16 of the M rows pick one expert).
   CTA `i` takes items `i, i+48, ...`. Every item costs the same bytes, so a round
   finishes together; at c1 33 CTAs stream one expert each, at c4 48 CTAs stream two.
2. **Weight-major streaming.** Each item streams its expert's weights exactly once, as
   one uniform 11.25 KB stage sequence through a 5-slot cp.async ring (4 stages, 45 KB,
   in flight per SM):
   - FC1: 40 channel blocks x (8 up rows, then the same 8 gate rows) = 80 stages; each
     stage is 8 full rows = one contiguous 10 KB span plus the rows' K64 scale words from
     the F8_128x4 plane. The 4 warps split K (10 K64 slices each) and reduce their FP32
     partials in a fixed order. The first version streamed 32 rows x 320 B quarter-rows
     at 1280 B stride and plateaued at 185 GB/s like dynamic: LPDDR row locality.
   - FC2: 40 blocks of 64 `down` rows x the full K=320 = 40 stages; each is 10 KB fully
     contiguous plus 64 x 5 scale words.
   The ring runs across the FC1->FC2 turn and into the next item, because weight
   addresses depend only on the expert id. Scale sectors shared by neighbouring 32-row
   groups are read again a few stages later from L2, so DRAM reads each byte once.
3. **Fusion.** Per item: gather + quantize the pass rows to NVFP4 in smem (a1 scale,
   `quantize_block_fp4`, overlapped with the ring prologue); FC1 with the native
   `m16n8k64 kind::mxf4nvf4` QMMA (A = the 16 pass rows, B = weight rows, FP32
   accumulate, K ascending); SiLU(alpha g) x (alpha u) in registers as soon as a
   channel block's gate finishes, rounded to BF16 in smem; per-16 NVFP4 requantize
   with the a2 scale; FC2 with the same QMMA over K ascending; `down_alpha x router
   weight`, then a BF16x2 atomic scatter-add into the output. One partial per expert:
   10 atomic adds per output element instead of 30.
4. **CUDA graphs.** A fixed grid, no host reads, no allocation, and all state in smem.
   The launcher zeroes the output in-stream before the launch (a capturable memset).

Numerics match `dynamic` step for step: the same input quantizer and scale, the same
QMMA instruction and fragment/scale mapping (from `nvfp4_phase1`), FC1 K order, the
alpha placement, the SiLU formula (`rcp_approx(1+exp(-g))`, same fast_math), BF16
rounding before the same per-16 requantizer, FC2 K order within an expert, and the same
BF16 atomic combine. The only difference is that the expert partial is not split into
three BF16-rounded slice partials. That can only reduce rounding, and the combine
order of the atomics is nondeterministic in both kernels.

## Bandwidth model and expected time

`t_layer = D x 1.3824 MB / B_eff + t_fixed`, with `t_fixed` ~ 8 us (launch/ramp,
routing + quantize prologue, and the output memset). The ring keeps 45 KB in flight
per SM, so B_eff should stay at about 240 GB/s (96% of 249) with 12 or more CTAs active.

| | D | bytes | t_layer (240 GB/s) | now | saved/layer | saved/step (48 layers) | step |
|---|---:|---:|---:|---:|---:|---:|---:|
| c1 | 33 | 45.6 MB | 198 us | 273 us | 75 us | **3.6 ms** | 43.3 -> 39.7 ms (tok/s +9%) |
| c4 | 96 | 132.7 MB | 561 us | 729 us | 168 us | **8.1 ms** | 74.4 -> 66.3 ms (tok/s +12%) |

c2 (D ~ 60) needs 12 CTAs in round 2 to pull ~21 GB/s each, which is within the ring's
reach. The band is M <= 32 (c1..c6 with 4 MTP drafts; `B12X_MOE_WM_MAX_TOKENS`); c8+
stays on `dynamic`.

## Selection

`B12X_MOE_DECODE_BACKEND=wm` puts `decode_backend=wm` into the MoE query controls, so
selection-cache keys differ only when it is set and the default path is byte-identical.
Eligible NVFP4/SiLU token counts <= the band then use `wm` without a race. vLLM declares
the same variable in `vllm/envs.py`, so the torch AOT compile key changes with it.

## Risks

- Per-SM streaming rate in the tail (few CTAs active) is unmeasured on GB10; the
  microbench measures it, and a channel-split item variant is the fallback.
- A second pass re-reads an expert's weights when > 16 rows pick it (rare at M <= 20).
- 1 CTA/SM with ~95 KB smem leaves no room for a co-resident aux-stream kernel on busy
  SMs. That is the same as `dynamic`, and at c1 the 15 idle CTAs exit at once.
