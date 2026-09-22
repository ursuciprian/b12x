# Review: GDN decode head-grouping work (results/codex-gdn, validation/gdn)

Read-only review of the uncommitted working tree in this clone (b12x @ a8333658,
serving version; remote `dgx-01:~/GEN-AI/build/b12x`). No GPU runs, no commits.

## 1. What was measured

Harness: `benchmarks/benchmark_gdn_decode.py --mode graph`, new cases
`qk8-v24-verify5-bs{1,4,8,10,16}` = Qwen3-Next GDN geometry (key_heads 8,
value_heads 24, head_ratio 3), FP32 state, all requests 5 query tokens
(4 MTP proposals + target), `--capacity-seqs 16 --capacity-columns 5`.
Single GB10 (dgx-01, SM 12.1, persistence on, SM clock 2184 MHz before and
after every run). 8 JSON files = 4 arms (`--head-group-size 0/1/2/3`) with
repeats at 100/300/500 iterations after 10/50/100 warmup.

Methodology, as read from the code rather than the prose:

- Timed region is `graph.replay()` only. `bench_cuda_graph`
  (`benchmarks/common.py:279`) records start/mid/end events around
  `prepare()` then `replay()`, so the state restore lands in `metadata_us`
  (reported as `restore_median_us`, ~1.01-1.04 ms/iter) and is **excluded**
  from the `median_us` numbers below. Correct.
- **L2 is not flushed** (`provenance.l2_flush = false`). It is not a real
  problem here: `prepare()` rewrites the whole 121 MiB state pool between
  every replay, which evicts far more than L2 holds.
- Per-replay hygiene is checked, not assumed: stable output/state/scratch
  addresses, zero replay allocation, output poisoned to NaN then compared
  against the FP32 torch reference (`graph_replay_after_output_poison: true`,
  `replay_allocation_bytes: 0`, `stable_addresses: true` in every report).
- Reported metric is the median of 100-500 replays plus min and p90; raw
  `samples_us` are kept in the JSON. Good evidence quality.

Medians (us), best repeat per arm, and the spread across repeats of the same arm:

| case | hgs=0 (default) | hgs=1 | hgs=2 | hgs=3 | own-arm spread (hgs=0) |
|---|---:|---:|---:|---:|---:|
| bs1  |  57.28 |  57.41 |  53.82 |  58.59 | 57.28-60.42 (5.5%) |
| bs4  | 182.62 | 193.63 | 184.61 | 191.46 | 182.62-185.34 (1.5%) |
| bs8  | 349.31 | 374.85 | 353.28 | 361.79 | 349.31-353.28 (1.1%) |
| bs10 | 434.14 | 465.55 | 438.61 | 446.46 | 434.14-437.46 (0.8%) |
| bs16 | 685.06 | 736.99 | 690.37 | 700.45 | 685.06-689.58 (0.7%) |

Correctness is identical in all arms (output max-abs 6.0e-08 at bs1,
3.9e-03 at bs>=8 against the FP32 reference; state max-abs 3.0e-08).

`validation/gdn/deferred_checkpoints.py` is a CPU-only FP32 oracle, not a
kernel: it shows that recording the per-token (delta, key, decay) and
replaying only the accepted prefix reproduces every accepted-prefix state
bit-exactly (60 checks, tokens in {1,2,5,8} x scale in {0.01,1,10}), and
models the state traffic as 2.531 -> 1.310 GiB/rank/step at c8. It ran here
on the Mac and passes. It explicitly disclaims GPU implementation,
allocator lifetime, prefix-cache interop and E2E speed.

## 2. What codex concluded

`_tuning.py` ships `qwen_head_group_size: (0,)` with the comment "Grouped
variants did not improve c8-c16; explicit research overrides only." The data
supports that: hgs=2 is inside its own arm's run-to-run spread at bs4-bs16
(+0.4% to +0.6% vs the best hgs=0 repeat, while hgs=0 itself varies 0.7-1.5%
between repeats), hgs=1 is 5-7% slower everywhere above bs1, hgs=3 is 2-5%
slower. Only bs1 shows a possible hgs=2 win (53.82-56.26 vs 57.28-60.42
median, 51.33 vs 54.24 min), and at c1 GDN is 2.3% of CUDA time, so even a
real 6% there is ~0.1% of the step. Negative result, correctly called.

## 3. Correctness and default-path risk in the diff

**Default path (`qwen_head_group_size=0`) is semantically unchanged.** Every
new constant collapses to the old one at hgs=0:
`grouped_value_heads = head_group_size if head_group_size > 1 else 2` -> 2
(old `_GROUPED_VALUE_HEADS`); `work_heads` -> `value_heads`, so `work_ctas`
and `total_work` are identical; the work-ID decode takes the `else` branch
(`value_head = work_head`, `key_head = value_head // head_ratio`) which is
the old code verbatim; `grouped_heads` keeps the old
`(end - start > 1) & (bounded_seqs > 1)` under `const_expr(head_group_size
== 0)`; `ungrouped_tail` compares against 2. The `const_expr` guards mean
hgs=0 compiles to the old control flow with no extra runtime branch. No
default-path risk from the kernel changes.

Notes on the non-default arms (they do not ship, but for the record):

- The compact mapping is only correct because `head_ratio` is hard-pinned to
  3 (`_cute_kernels.py:236`). For hgs=2 the per-key groups start at value
  heads 0 and 2; group 1 has `value_head_in_group == 2 == grouped_value_heads`
  so it is routed to the single-head `_run_request`. With any other ratio
  (e.g. 4) that last group would silently process one head and drop the
  rest. The ratio guard saves it today; `_validate_config` only rejects the
  KDA case (`key_heads == value_heads`), not a non-3 ratio.
- For hgs>1, `grouped_heads` is forced true, dropping the legacy
  `end - start > 1` and `bounded_seqs > 1` conditions. That is required by
  the compact work-ID mapping (no follower CTAs exist to pick up the slack)
  and `tests/sequence/test_gdn_head_groups_gpu.py` asserts bit-identity
  against the legacy dispatch for lengths `(5,)*{1,4,8,10,16}`,
  `(5,2,0,1,4)` and `()` — unverified here, it needs the GPU.

**Schema bump 4 -> 5: yes, it invalidates the plan cache and forces a
re-selection for `sequence.gdn_decode` on the next boot with this b12x.**
`_choice_key` (`b12x/preparation/session.py:780-787`) digests
`config_schema_version` and `candidate_contract_version`, both of which
changed (4->5 and 3->5), so every persisted selection for this component
misses. `_declaration_key` (line 118) digests them too, so shared-declaration
aliasing misses as well. Blast radius is bounded but non-zero:

- The candidate space is still a single point (`backend` one value,
  `qwen_head_group_size` one value), so the "retune" is one candidate — a
  compile and a measurement, not a search.
- Under `cache_only=True` the miss is not free: that path depends on the same
  key (`session.py:948`, `997`, `1494`) and on lowering being permitted
  (`_measurement.py`), so a cache-only boot pays the lowering it was meant
  to avoid for this component.
- `GdnConfig.from_config` now requires `qwen_head_group_size` in the payload
  and raises `ValueError` without it. Fail-closed is the right behaviour and
  the digest change means old entries are never looked up by key — but any
  pinned/exported GDN config loaded by path rather than by digest would hard
  fail. Worth checking before this ships anywhere near the serving recipe.

Conclusion: **do not ship the schema bump for a knob whose only shipped value
is 0.** If the grouping code is kept for research, keep it out of the
`config_fields`/schema (e.g. an env- or override-only compile parameter) so
the serving plan cache on dgx-01 is untouched.

Two smaller evidence-quality points in the harness:

- `build_case` now pins `override=GdnConfig(backend="cutedsl",
  recurrent_block_v=32, ...)` for *every* case, including the
  `--head-group-size 0` runs. The benchmark therefore no longer measures what
  tuning would select. For the Qwen cases the tuned choice is the same
  (cutedsl, and `recurrent_block_v=32` is mandatory for `key_heads !=
  value_heads`), so the numbers stand — but the "default" arm is a pinned
  config, not the production dispatch.
- `_reference` in both the benchmark and the tests now reaches into
  `binding._state.caps` instead of `binding.plan.caps`. Private-attribute
  reach-in from test/bench code; harmless, but it will rot.

## 4. CPU-only test results (run on the Mac, no GPU)

- `python3 -m unittest tests.sequence.test_gdn_head_group_contract` — **3
  tests, OK.** Covers: default config keeps `qwen_head_group_size == 0`; the
  candidate set is exactly `{0}`; all four values round-trip through
  `config_payload`/`decode_config`; KDA (`key_heads == value_heads`) rejects
  grouping; `True`, `1.5`, `-1`, `4` are rejected; a payload missing the new
  field is rejected.
- `python3 validation/gdn/deferred_checkpoints.py` — **PASS**, 60
  accepted-prefix states bit-identical, prints the 2.531 -> 1.310
  GiB/rank/step model.
- `tests/sequence/test_benchmark_gdn_decode.py` and
  `tests/sequence/test_gdn_head_groups_gpu.py` could **not** run: pytest is
  not installed in this Python (3.13.7) and `cutlass` is absent, so
  `benchmarks.common` will not import. Both files parse cleanly (`ast.parse`)
  and the new case-name assertion matches the new case generator by
  inspection. The GPU test needs dgx-01.

## 5. Verdict: the negative result is solid, and it points at the real lever

The grouping result is not just solid, it is explained. Compute the DRAM
traffic the kernel actually moves. `_run_request` and
`_run_grouped_request` write the **full state tile once per verified token**
(`_cute_kernels.py:511-541` and `775-810`, inside the `relative_token` loop,
each to `state_indices[request, relative_token]`). So per GDN layer per step:
one snapshot read plus **five** snapshot writes, where a snapshot is
`24 * 128 * 128 * 4 B = 1.5 MiB` per request.

| case | state bytes moved | median | effective GB/s | % of GB10 273 GB/s |
|---|---:|---:|---:|---:|
| bs1  |   9 MiB |  57.28 us | 165 | 60% |
| bs4  |  36 MiB | 182.62 us | 207 | 76% |
| bs8  |  72 MiB | 349.31 us | 216 | 79% |
| bs10 |  90 MiB | 434.14 us | 217 | 80% |
| bs16 | 144 MiB | 685.06 us | 220 | 81% |

The production trace agrees: the profiling README records 322.31 us mean for
`b12xsequencegdn_decode_cute_kernels_Pa...` at c8, which on the same 72 MiB
is 234 GB/s, 86% of peak. It also settles that README's open question —
3492 launches over 100 steps is ~35/step, i.e. **one launch per GDN layer
already covering all live requests**, not 8 per-sequence launches. The
"batch across requests" lever is already taken.

So: at c8-c16 this kernel is DRAM-bandwidth-saturated at ~80-86% of the
GB10's 273 GB/s. No CTA/head/work-ID geometry can help, which is exactly
what hgs=1/2/3 measured. The 1/3 of work IDs that hgs=0 wastes on no-op
follower CTAs cost nothing because they exit before touching memory, and
removing them (hgs=2) buys nothing. hgs=1 is *slower* precisely because it
stops sharing the q/k L2-norm through shared memory and adds read traffic.
There is no grouping, tiling or verification-length variant left to try:
the only lever is bytes.

**The lever codex found and then left unimplemented is in
`validation/gdn/`.** Those five per-token checkpoint writes are 5/6 of GDN's
traffic, and they exist only because acceptance is not known until after the
sampler. The oracle proves the accepted-prefix state is recoverable
bit-exactly from the initial state plus tiny per-token records
(`5 * (24*128 + 8*128 + 24) * 4 B` per layer, ~0.4% of a snapshot). Traffic
then drops from 6 snapshots to 3 (verify read, commit read, commit write),
the 1.93x the script models. Since the kernel is ~80-86% bandwidth-bound,
time should track traffic: GDN c8 13.5% of step -> ~7%, i.e. **~6.5% of the
c8 step**, matching the ~7% this was worth on paper.

Two things worth noting beyond the script's own model:

1. If the commit replay is **fused into the next step's verify pass** (which
   must read the state anyway), the steady state becomes one snapshot read
   and one small record write per step — up to 6x less traffic, GDN c8 ->
   ~2-3% of step. Full snapshot writes would only be needed when a sequence
   finishes or its slot is evicted. That is the version to aim at.
2. It also shrinks the state pool: the per-column slots
   (`state_index_columns = num_spec + 1 = 5`) exist to hold those
   per-token checkpoints. Records instead of slots frees most of the GDN
   state pool for KV or more concurrent sequences.

Unvalidated, and in rough order of risk: the CuTe implementation of the
two-pass (or fused) scheme, the FP32 multiply/add contraction the oracle
pins, allocator/CUDA-graph lifetime for the record buffers under stable
addresses, interaction with the vLLM mamba state-pool and prefix caching,
and E2E quality at the gated `--levels 1 >= 95 tok/s` bar.

## Recommendation

1. Do not ship `_tuning.py` as it stands: the `config_schema_version` /
   `candidate_contract_version` bumps force a plan-cache miss and
   re-selection for `sequence.gdn_decode` on dgx-01 in exchange for a knob
   that only ever takes the value 0. Either drop the grouping knob from the
   tuning schema entirely (keep it as a research-only compile override) or
   land it together with something that uses it.
2. Keep the benchmark cases (`qk8-v24-verify5-bs*`), the contract test and
   `validation/gdn/` — they are the useful output of this pass. Drop the
   pinned `override` in `build_case` when `--head-group-size 0` so the
   default arm measures the production dispatch again.
3. Next GDN work should attack the per-token checkpoint writes, not the head
   geometry. Start by confirming on dgx-01 that the serving path really does
   hand the kernel 5 distinct destination slots per request (it should, given
   `state_index_columns = 5`), then prototype the deferred-record commit —
   ideally fused into the following step's verify pass.
