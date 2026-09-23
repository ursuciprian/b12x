# Deferred GDN decode checkpoints

Status: implemented in b12x behind `Caps(deferred_checkpoints=True)`, **default
off**, branch `feat/gdn-deferred-checkpoints` from `a8333658`. The vLLM side is
implemented too (§6) but not enabled anywhere: it needs a b12x wheel carrying
this branch, and no published wheel has one. Nothing is measured yet.

## 1. Why

`results/codex-gdn/REVIEW.md` settles that Qwen GDN decode is DRAM-bandwidth
saturated at c8-c16, at 79-86% of the GB10's 273 GB/s. No CTA, head or work-ID
geometry can help, and head grouping measured exactly that (negative result,
`research/gdn-head-groups`). The only lever left is bytes.

Where the bytes go: `_run_request` and `_run_grouped_request`
(`b12x/sequence/gdn_decode/_cute_kernels.py`) persist the **full recurrent-state
tile once per verified token**, to `state_indices[request, relative_token]`.
Per GDN layer per request per step, at MTP 4:

| | snapshots |
|---|---|
| read the initial checkpoint from column `accepted - 1` | 1 |
| write a post-token checkpoint for each of 5 verified tokens | 5 |

Five of those six snapshots exist only because acceptance is not known until
after the sampler. Four of the five are thrown away.

`validation/gdn/deferred_checkpoints.py` proves the way out on CPU: recording
the per-token `(delta, key, decay)` and replaying only the accepted prefix onto
the pre-verify state reproduces **every** accepted-prefix state bit-exactly (60
checks, tokens in {1,2,5,8} x scale in {0.01,1,10}).

## 2. The scheme

Two changes to one kernel, and no new buffer anywhere.

**Verify pass** (the existing decode kernel, per request):

1. Read the base snapshot from `state_indices[request, 0]` — one snapshot read,
   which the kernel already does.
2. **Commit, fused into that read**: replay records for tokens
   `1 .. accepted - 1` of the *previous* step onto the register-resident state.
   The result is bit-identical to the checkpoint the shipped kernel wrote into
   column `accepted - 1`.
3. Run the verified tokens as today. At `relative_token == 0`, write the full
   snapshot to column 0 exactly as today. At `relative_token >= 1`, write a
   compact record instead of a snapshot.

The commit is fused rather than a separate kernel because the verify pass reads
the state anyway: fusing costs one snapshot read+write per step instead of the
two reads and two writes a standalone commit-then-verify pair would cost (§4).

**No validity flag is needed, and that is the point of anchoring the base at
column 0.** The number of records replayed is exactly `accepted - 1`, and
records `1 .. accepted - 1` are written by the same step that wrote column 0.
The one case with no records — a request whose previous step was a prefill — is
exactly the case where `accepted == 1`, so zero records are replayed. This is
not a new invariant: the shipped kernel already requires that column
`accepted - 1` was written by the previous step.

### Where the records live

**Inside the speculative state slots the scheme stops using.** Record for token
`j` goes into the slot `state_indices[request, j]`, `j = 1 .. columns-1`, at
offset 0. Those slots are exclusively-owned, unhashed, never prefix-cached
scratch (`vllm/v1/core/single_type_kv_cache_manager.py:2280-2289` on dgx-01),
and under this scheme nothing else defines their contents.

Consequences, all good:

- **Zero new allocation.** No workspace slice (the b12x scratch arena is reused
  across ops within a step — `vllm/v1/worker/workspace.py:181-206` — and is
  locked after graph capture, so it could not have held records anyway), no new
  staging tensor, no `PersistentMemory` key, no new kernel argument.
- **Addressed by the existing `state_indices`**, so batch reordering between
  steps is irrelevant: row `request` is looked up fresh each step for the
  current request. A record buffer keyed by the batch row would have been
  silently misattributed when a request finishes and the compacted spec index
  shifts.
- **CUDA-graph-safe for free**: the record storage is `self.kv_cache[1]`, whose
  address is already stable across replays.
- Lifetime is the existing one: columns `0 .. num_spec` must hold what the
  previous step wrote, which is already what the shipped kernel depends on.

### Record layout (compile-time constants)

One record block per `(request, value_head, value_tile)`, i.e. per work item, so
**the only reader of a block is the work item that wrote it** and within a
kernel invocation its replay reads precede its record writes. No cross-CTA
ordering, no ping-pong, no grid barrier.

```
block at:  state_indices[request, j] * state_slot_stride
           + (value_head * 4 + value_tile) * 176        # floats
  +   0 .. 127   key     the FP32 L2-normalized key, exactly as consumed
  + 128 .. 159   delta   one per value row of the tile
  + 160          decay   scalar
  (176 floats = 704 B, every field 16-byte aligned)
```

`_VALUE_TILES * _RECORD_FLOATS = 4 * 176 = 704 <= _VALUE_DIM * _KEY_DIM =
16384`, asserted at module import: a record block always fits inside one value
head of a state slot, for any head count.

The key is replicated per value head rather than per key head (3x for Qwen's
`head_ratio = 3`). Keying it per key head would make three work items share one
slot, which reintroduces a write-before-read race across CTAs and would need a
double-buffered record region plus a device parity flag. The replication costs
~5% more traffic than the theoretical minimum and buys away all of that.
`ponytail: 3x key replication, ping-pong the record region if the last 5% of
GDN traffic ever matters.`

### Memory contract

| buffer | owner | lifetime | changed? |
|---|---|---|---|
| `recurrent_state` columns `[r, 0]` | caller pool | across steps | now holds the state **after token 0**, not the committed state |
| `recurrent_state` columns `[r, 1..num_spec]` | caller pool | one step (written at N, read at N+1) | now hold records, not checkpoints |
| `state_indices`, `num_accepted_tokens`, `num_seqs`, `query_start_loc` | caller | per step | unchanged |
| scratch | caller | per call | still 0 bytes |

Two preconditions the shipped path tolerated and this one does not. **Active
state-index cells must be unique**: a duplicate used to mean a checkpoint
written twice, and now means a record overwriting the base state. And the
**slot stride must hold a record block** (`value_heads * 4 * 176` elements),
which `_binding_key` checks against the bound tensor rather than asserting over
compile-time constants.

**Accepted length arrives as a device tensor**, `num_accepted_tokens`
(`int32[max_seqs]`), which the kernel already reads to pick its source column.
On the vLLM side the post-sampler value is device-only —
`self.num_accepted_tokens.gpu`, whose CPU mirror is not synchronized until the
next step's `_prepare_inputs` (`gpu_model_runner.py:2217-2222`) — so nothing in
this scheme needs a host read. That is why the commit is expressed as a replay
length read from device memory rather than a launch scalar.

Under CUDA-graph replay nothing new needs stable addresses: the kernel takes no
new pointer and no new scalar. The knob is a compile-time constant, so the
deferred and shipped paths are different compiled programs, never a runtime
branch.

### Numerics

Bit-identical to today's accepted-prefix checkpoint. The replay applies, per
state element, in this order:

```python
state_value = mul_rn_f32(state[e], decay)          # the stored FP32 decay
state[e]    = fma_rn_f32(delta, key, state_value)  # the stored delta and key
```

Those are inline-asm `mul.rn.f32` and `fma.rn.f32`, because the *contraction*
is part of the contract, not just the value. The verify loop's decayed value
has a second use — the `state_dot_k` accumulation — so it is materialized and
the update is `fma(delta, key, rn(state * decay))`. Written as plain
`a * b + c * d` the replay's decayed value would have one use, and folding the
**left** multiply instead is just as valid: that gives
`fma(state, decay, rn(delta * key))`, a different FP32 number on ~99% of random
elements, which `test_the_two_fma_orders_are_observably_different` measures
directly. Pinning both instructions removes the compiler's choice. The
stored `decay` is the `exp(-exp(A_log) * softplus(a + dt_bias))` the verify pass
computed; the stored `delta` is `(value - state_dot_k) * beta` **after** the
BF16 rounding Qwen applies to beta; the stored `key` is the L2-normalized FP32
key as the state update consumed it (not the raw `mixed_qkv` value). Nothing is
recomputed, so there is nothing to diverge.

Guarded by construction, and rejected at `Caps` and `GdnQuery` validation:
FP32 state only (BF16 records would not round-trip), Qwen heads only (not KDA,
which uses the Triton recurrence), and `null_state_index=None` — the serving
decode caps already satisfy all three
(`qwen_gdn_linear_attn.py:841-859` on dgx-01).

## 3. Traffic model

Per request per GDN layer per step, FP32 state, 24 value heads, MTP 4
(`snapshot = 24*128*128*4 = 1.5 MiB`, `record = 24*4*704 = 66 KiB`):

| | shipped | deferred |
|---|---|---|
| snapshot reads | 1 | 1 |
| snapshot writes | 5 | 1 |
| record writes | – | 4 |
| record reads | – | accepted - 1 |
| **bytes** | **9.00 MiB** | **3.26 / 3.39 / 3.52 MiB** at accepted 1 / 3 / 5 |

Per rank per step, 36 GDN layers (`python3 validation/gdn/deferred_checkpoints.py`):

| | c1 | c8 | c16 |
|---|---|---|---|
| shipped | 0.316 GiB | 2.531 GiB | 5.063 GiB |
| deferred (accepted 3) | 0.119 GiB | 0.953 GiB | 1.906 GiB |
| ratio | 2.66x | 2.66x | 2.66x |

Note the oracle's original 1.93x modelled a *standalone* commit kernel (verify
read, commit read, commit write = 3 snapshots). Fusing the commit into the
verify read gets it to 2 snapshots and 2.66x. `REVIEW.md`'s "up to 6x" was too
optimistic: the committed state has to be written back once per step to bound
the record chain, so two snapshots is the floor for this scheme.

### Expected time

Measured: 322 µs per GDN layer-step at c8 moving 72 MiB, i.e. 234 GB/s = 86% of
273 GB/s peak; GDN is 13.5% of the c8 step, so 36 layers x 322 µs = 11.6 ms of
an ~86 ms step. Deferred moves 27.1 MiB. At the same 234 GB/s that is 116 µs;
allowing for a smaller footprint sustaining less bandwidth, and for the replay's
up-to-four extra rank-one updates on top of five verify tokens (arithmetic on a
memory-bound kernel, largely hidden), the honest band is **130-160 µs**.

| | per layer-step | GDN share of the c8 step | step time |
|---|---|---|---|
| shipped | 322 µs | 13.5% | 86 ms |
| deferred, optimistic | 130 µs | 5.9% | -8.1% |
| deferred, conservative | 160 µs | 7.2% | -6.8% |

That meets the `<= 7%` GDN-share target at the optimistic end and sits on it at
the conservative end. At c1 GDN is 2.3% of the step and the kernel is
latency-bound at 60% of peak, so the c1 gain is under 1% of the step — not the
point. c16 should track c8 (2.1-2.4x on a kernel at 81% of peak).

**Unmeasured until the recipe below runs on dgx-01.** Everything above is a
model.

## 4. What is implemented

All of it is behind `Caps(deferred_checkpoints=True)`, default `False`.

- `_lib/intrinsics.py` — `fma_rn_f32` / `mul_rn_f32`, single PTX
  instructions behind opaque inline asm. The replay uses them because the
  contraction, not just the value, is part of the contract: see Numerics.
- `_cute_kernels.py`
  - `_record_base`, `_store_deferred_record`, `_replay_deferred_records` —
    module-level `@cute.jit` helpers, shared by the verify and commit kernels.
  - `_PackedRecurrentQwenKernel(deferred_checkpoints=...)`: the source column
    becomes 0, the replay runs after each state load in both `_run_request` and
    `_run_grouped_request`, and the token loop writes a record instead of a
    snapshot for `relative_token >= 1`. Every new branch is
    `cutlass.const_expr`-guarded and `relative_token` is a `range_constexpr`
    Python int, so with the knob off every one of them folds away at trace
    time and the emitted program is semantically the old one. (It is not
    textually the old one: both request helpers gained an unused
    `accepted_column` parameter and `source_column` is an extra SSA name that
    aliases it.) `head_ratio` pinning and the KDA path are untouched.
  - `_CommitDeferredQwenKernel` + `run_commit_deferred_checkpoints` — the
    standalone commit: read column 0, replay `accepted - 1`, write to a
    per-request destination slot. Needed because column 0 is no longer the
    committed state, so any reader outside the decode step must materialize it.
  - `_binding_key` appends `"deferred_checkpoints"` **only when the knob is on**,
    so the shipped configuration keeps a8333658's compiled-program identity.
- `_impl.py` — `Caps.deferred_checkpoints` (+ validation),
  `commit_deferred_checkpoints(binding, destination_indices)`.
- `_tuning.py` — `GdnQuery.deferred_checkpoints`, validation, and the knob
  **excluded from `_KEY_FIELDS`**, exactly like `max_state_slots`.
- `_preparation.py` — the flag threads `Caps` -> `GdnQuery` -> `compile_decode`.
- `api.py` / `__init__.py` — `commit_deferred_checkpoints` exported.

### Deliberately not a tuning knob

`query_schema_version`, `config_schema_version` and
`candidate_contract_version` are **unchanged (4 / 4 / 3)**, and
`config_fields` is still `{backend, recurrent_block_v}`. The knob is excluded
from `_KEY_FIELDS`, so `TUNING.encode_query` is byte-identical with the knob on
or off, so `_choice_key` (`b12x/preparation/session.py:780-787`) and
`_declaration_key` (`:118`) are unchanged and **no persisted selection for
`sequence.gdn_decode` misses on the next boot on dgx-01**. That is the specific
failure `results/codex-gdn/REVIEW.md` §3 flagged in the head-grouping diff, and
`tests/sequence/test_gdn_deferred_checkpoints.py::test_the_knob_stays_out_of_the_selection_key`
is the regression guard.

It is legitimate to keep it out of the selection key: the candidate space is a
single point either way, the knob changes the memory contract rather than which
configuration is fastest, and the compiled-program identity carries it.

### Not done

- **Shrinking the pool.** Columns `1 .. num_spec` now hold 66 KiB of records
  inside a 1.5 MiB slot. Making the speculative blocks small would free
  `4 * 1.5 MiB * max_seqs * 36` — 3.4 GiB at `max_seqs=16` — for KV or
  concurrency. That is a vLLM allocator change (`MambaSpec.page_size_bytes` /
  `max_memory_usage_bytes`, `vllm/v1/kv_cache_interface.py:965-976`), not a
  kernel change, and it is the obvious follow-up.
- Per-key-head records + a double-buffered record region (see §2).
- Anything on the KDA/Triton recurrence.

## 5. Validation

The whole recipe, one line:

```bash
ssh dgx-01 'cd ~/GEN-AI/build/b12x && git fetch fork && git checkout feat/gdn-deferred-checkpoints && pytest tests/sequence/test_gdn_deferred_checkpoints.py -q && python3 benchmarks/benchmark_gdn_decode.py --mode graph --cases qk8-v24-verify5-bs8 --deferred 0 1'
```

Sweep the batch sizes with
`--cases qk8-v24-verify5-bs1,qk8-v24-verify5-bs4,qk8-v24-verify5-bs8,qk8-v24-verify5-bs16`.

What each half proves:

- `tests/sequence/test_gdn_deferred_checkpoints.py` — bit-identity, **kernel
  against kernel at `rtol=0, atol=0`** (a torch reference reduces in a different
  order and cannot settle a bit-identity claim). Two plans over identical
  inputs, shipped and deferred:
  - `test_commit_reproduces_every_accepted_prefix_checkpoint` — one verify step,
    then for accepted 1..5 the standalone commit must reproduce the shipped
    kernel's column `accepted - 1`. This is the CPU oracle's claim, on the GPU.
  - `test_the_next_step_replays_the_accepted_prefix_in_place` — two steps: the
    second step's output and committed state must match the shipped path for
    accepted 1..5. This covers the *fused* replay, which is the path that ships.
  - Both at bs 1 / 8 / 16, which exercises the ungrouped (bs1) and grouped
    (bs>1 with multi-token requests) dispatches.
  - `test_mixed_accepted_and_query_lengths_stay_per_request` — accepted
    `(1,5,3,2)` over query lengths `(1,5,3,2)`. Uniform cases cannot see a
    replay length borrowed from the wrong request.
  - `test_the_replay_matches_the_verify_fma_contraction` — a large state
    with keys near one, so the two contraction orders land an ULP apart and the
    rtol=0 comparison is actually load-bearing.
  - `test_a_destination_aliasing_a_record_column_is_refused` — pointing the
    commit at one of the request's own record blocks must leave the pool
    untouched, not race the CTAs still replaying out of it.
  - `test_a_single_column_plan_needs_no_records` — a one-column plan must be
    identical with the knob on.
  - The contract tests run without a GPU and are the plan-cache regression guard.
- `benchmarks/benchmark_gdn_decode.py --deferred 0 1` — `0` is the shipped arm,
  unchanged, still gated against the torch reference. `1` primes the record
  columns, gates the verify half against the reference and the commit half
  against the reference's accepted column (`deferred_commit_max_abs`, at the
  FP32 reference tolerance -- bit identity is pytest's job), then freezes the primed pool as the restore image so every timed
  replay measures the steady state: base read + replay, then base write +
  record writes. The reference cannot read a pool whose speculative columns
  hold records, so the graph-replay gate feeds it the logical accepted-prefix
  state (the commit applied to a copy of the restore image, placed in column
  `accepted - 1`) and then checks the output and, for every accepted length
  1..L, the committed state of the replayed pool against the reference's full
  checkpoint for that token. Raw record columns are never compared with
  checkpoint columns.
- `test_graph_replays_follow_device_live_counts_acceptance_and_order` — one
  capture per path, then replays with device-side live counts 0/1/4/8/10/16,
  mixed lengths, accepted 1..5 and permuted row order, under
  `kernel_resolution_guard`: bit-identical to the shipped path, no allocation,
  no new CuTe cache entries, unchanged data pointers, idle windows untouched.
- `test_replay_and_commit_address_slots_beyond_two_to_the_31` — the same plan
  over a compact pool and a pool whose slot stride puts slots 5..10 past
  element 2^31 (~16 GiB). Opt in with `B12X_TEST_LARGE_POOL=1` in an exclusive
  GPU window; it skips on insufficient free memory.

CPU-side, on any box: `python3 -m py_compile` over the touched files and
`python3 validation/gdn/deferred_checkpoints.py` (60 bit-identical
accepted-prefix states, plus the traffic model above).

Do **not** enable this in serving without the vLLM side of §6: with the knob on
and an unpatched vLLM, the block-boundary state copy would copy a record block.

## 6. §vLLM

**Implemented, not sketched.** `ursuciprian/vllm` branch
`feat/gdn-deferred-checkpoints`, commit `29bf8477f`, cut against `8e1f1e587f`
(the image base). Every line number below is against `8e1f1e587f`; the earlier
revision of this section cited a different tree and got most of them wrong.
The sparkrun mod that applies it to an installed image is
`mods/vllm-gdn-deferred/` in the `qwen3.8-flash-next` registry.

### The readers of a speculative column

Four, not the two the first draft of this section listed:

| reader | file:line | handled how |
|---|---|---|
| same-step boundary copy | `vllm/v1/worker/mamba_utils.py:557` in `postprocess_mamba_fused_kernel` (`:444`) | decide → commit → copy |
| next-step boundary copy | `:688` in `precopy_mamba_align_fused_kernel` (`:627`) | decide → commit → copy |
| boundary checkpoint export | `:419` in `checkpoint_mamba_states_kernel` (`:379`), reached from `vllm/v1/worker/gpu/boundary_checkpoint.py:651` `capture_mamba` → `checkpoint_request_boundaries` (`mamba_utils.py:1393`) | **refused** |
| CPU-metadata fallback | `collect_mamba_copy_meta` (`:1545`), reached from `preprocess_mamba:1749` | **refused** (non-fused path) |

All three kernel readers go through one shared helper, `_copy_mamba_state_block`
(`:197-374`), whose docstring (`:225-229`) states both contracts. That helper
does the conv **and** temporal state from a single `token_bias`, which is why
"keep the conv bias, make the temporal half a commit" needed the helper split
first — there is no per-half call site to redirect.

### The edits

1. **`vllm/envs.py`** — `VLLM_GDN_DEFERRED_CHECKPOINTS`, default off.

2. **`vllm/v1/worker/gdn_deferred_commit.py`** (new) — the refusal rules, the
   per-step commit buffers, and `gather_gdn_commit_windows_kernel`, which turns
   a request's pre-advance running column into the `[r, 0..num_spec]` window
   b12x's commit expects. Nothing here is reached unless the flag resolves on.

3. **`mamba_utils.py:197`** — `token_bias` becomes `conv_bias` + `temporal_bias`.
   The conv halves (`:282`, `:290`, `:321-331`, `:347`) keep the accepted-token
   bias; the temporal half (`:359-362`) takes the new one. All three call sites
   pass both.

4. **`mamba_utils.py:444` and `:627`** — each gains `DEFERRED_TEMPORAL` (pass
   `temporal_bias = 0`, because the commit already replayed into `bt[src_col]`)
   and `DECISION_ONLY` (emit `commit_src_col` / `commit_accepted` and return
   without copying). `DECISION_ONLY` is what keeps the copy decision in exactly
   one place: the drivers launch the same kernel twice rather than
   reimplementing `aligned_new_computed >= num_tokens_running_state` (`:537`).

5. **The two drivers** — `run_fused_postprocess_align` and `run_fused_precopy`
   run decide → commit → copy when the feature is active, and pass
   `DEFERRED_TEMPORAL` to the real copy.

6. **`preprocess_mamba:1730-1753`** — the assertion §6(d) promised: a request's
   window must hold distinct blocks. Under this mode a duplicate cell stops
   being a redundant checkpoint write and becomes a record overwriting the base
   state, so uniqueness is now correctness, not freshness. It is a handful of
   host comparisons per step. The accepted-count reset to 1 that the commit
   contract needs is already there (`:550` device-side, `:1753` host-side).

7. **`qwen_gdn_linear_attn.py`** — `_initialize_b12x_gdn_decode` (`:813`)
   resolves the flag; `_make_b12x_gdn_caps` (`:841`) declares
   `deferred_checkpoints=`; `_bind_b12x_gdn_decode` (`:1222`, binding the pool
   at `:1247`) accepts metadata overrides, because the commit runs outside the
   forward pass where the staged metadata still describes the previous step;
   and `commit_b12x_gdn_deferred` is the per-layer entry point. The flag is
   read next to the other GDN options in `additional_config` (`:272`).

### Refused, deliberately

`refuse_reasons` names each one instead of silently falling back, because the
two paths differ in what the state pool *means* and a silent downgrade would be
invisible in a benchmark:

- **Request-boundary checkpoints.** One `capture_mamba` can carry up to three
  distinct biases for the same request (prompt, response and instruction
  slots, `boundary_checkpoint.py:128-130`), and a single accepted-prefix commit
  produces one state. Exporting a record block as a state would poison the
  checkpoint and the prefix-cache entry built from it.
- **Non-align cache mode**, where the speculative columns are not exclusively
  owned.
- **Zero speculative tokens**, where there is nothing to defer.

### Why the pool is safe

- Speculative blocks are exclusively owned and unhashed
  (`single_type_kv_cache_manager.py:2285`, from `:2226`), so no other request
  ever observes a record block.
- `_relocate_speculative_block` (`:2280`) only re-slots within one request's own
  table, so a record written at step N is still addressable at step N+1 — the
  same property the shipped kernel already relies on to read
  `state_indices[r, accepted - 1]`.
- Only the running/aligned block is prefix-cached.
- Sizing is untouched: `MambaSpec.page_size_bytes` (`kv_cache_interface.py:955`)
  and `max_memory_usage_bytes` (`:965`) still allocate
  `2 + num_speculative_blocks` blocks. Shrinking the speculative blocks to the
  66 KiB a record actually needs would free ~3.4 GiB at `max_seqs=16`, and is
  the obvious follow-up.
