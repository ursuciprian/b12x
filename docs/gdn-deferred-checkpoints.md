# Deferred GDN decode checkpoints

Status: implemented in b12x behind `Caps(deferred_checkpoints=True)`, **default
off**. Not yet enabled in the vLLM fork; §vLLM is a patch sketch, not applied.
Branch `feat/gdn-deferred-checkpoints`, from `a8333658`.

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
state_value = state[e] * decay          # the stored FP32 decay
state[e]    = state_value + delta * key # the stored FP32 delta and key
```

which is the same two statements, in the same shape, as the verify loop's
`state_value = state[e] * decay` followed by `state_value = state[e] + delta *
shared_k[...]`. Same operands, same FP32 contraction, same FMA selection. The
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

- `_cute_kernels.py`
  - `_record_base`, `_store_deferred_record`, `_replay_deferred_records` —
    module-level `@cute.jit` helpers, shared by the verify and commit kernels.
  - `_PackedRecurrentQwenKernel(deferred_checkpoints=...)`: the source column
    becomes 0, the replay runs after each state load in both `_run_request` and
    `_run_grouped_request`, and the token loop writes a record instead of a
    snapshot for `relative_token >= 1`. Every new branch is
    `cutlass.const_expr`-guarded and `relative_token` is a `range_constexpr`
    Python int, so with the knob off the traced program is the old one
    statement for statement. `head_ratio` pinning and the KDA path are untouched.
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
  - `test_a_single_column_plan_needs_no_records` — a one-column plan must be
    identical with the knob on.
  - The contract tests run without a GPU and are the plan-cache regression guard.
- `benchmarks/benchmark_gdn_decode.py --deferred 0 1` — `0` is the shipped arm,
  unchanged, still gated against the torch reference. `1` primes the record
  columns, gates the verify half against the reference and the commit half
  against the reference's accepted column (`deferred_commit_max_abs`, at the
  FP32 reference tolerance -- bit identity is pytest's job), then freezes the primed pool as the restore image so every timed
  replay measures the steady state: base read + replay, then base write +
  record writes. The reference cannot model a pool whose speculative columns
  hold records, which is why the deferred arm's graph-replay comparison is
  skipped and the numerics live in pytest.

CPU-side, on any box: `python3 -m py_compile` over the touched files and
`python3 validation/gdn/deferred_checkpoints.py` (60 bit-identical
accepted-prefix states, plus the traffic model above).

Do **not** enable this in serving before §vLLM lands: with the knob on and the
old vLLM binding, the block-boundary state copy would copy a record block.

## 6. §vLLM

Fork: `~/GEN-AI/build/vllm` on dgx-01, branch `dev/jovian-judgement`, HEAD
`e624ae19a3`. Line numbers are as they exist there now. **Sketch, not applied.**

The decode call path needs no new tensor: `api.run(self._bind_b12x_gdn_decode(...))`
at `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:2615-2625`
already passes `recurrent_state=self.kv_cache[1]`,
`state_indices=staging.state_indices` and
`num_accepted_tokens=staging.num_accepted_tokens`, which is everything the
scheme uses. Four edits:

**(a) Declare the contract.** `_make_b12x_gdn_caps`, `:841-859`:

```python
     caps = api.Caps(
         ...
         qk_l2norm=True,
+        deferred_checkpoints=self._gdn_deferred_checkpoints,
     )
```

with the flag resolved next to `gdn_decode_kernel` (`_resolve_gdn_decode_kernel`,
`:315`, reading `additional_config`, `:281`) as
`additional_config["gdn_deferred_checkpoints"]`, default false, and refused
unless `gdn_decode_kernel == "b12x"`, `state_dtype` is FP32 and
`num_spec >= 1`. The prefill caps (`:963`, which set `checkpoint_export=True`
and `null_state_index=0`) are untouched.

**(b) Replace the temporal block-boundary copy with the commit.** This is the
only place outside the decode kernel that reads a speculative column:
`_copy_mamba_state_block`, `vllm/v1/worker/mamba_utils.py:359-375`:

```
362	    actual_src_block_id = tl.load(block_table_base + src_col + token_bias).to(tl.int64)
363	    src_addr = state_base_addr + actual_src_block_id * state_block_stride
```

With deferred checkpoints, `block_table[src_col + token_bias]` is a record
block, not a state. The temporal half of that copy must become, per GDN layer:

```python
# deferred checkpoints: the accepted prefix is replayed, not selected
destination = torch.full((max_seqs,), -1, dtype=torch.int32, device=dev)
destination[boundary_requests] = block_table[boundary_requests, dst_col]
api.commit_deferred_checkpoints(layer_binding, destination)
```

`destination_indices` is `int32[sequence_capacity]`; `-1` skips a request, so
the tensor carries the `needs_copy` predicate that
`postprocess_mamba_fused_kernel` computes at `mamba_utils.py:537-540`. The
**conv** half of the copy keeps its `token_bias` window shift unchanged.

Three call sites read that helper and all three need the same treatment:
- `postprocess_mamba_fused_kernel`, `mamba_utils.py:444-577` (copy at `:557-577`),
  reached from `postprocess_mamba_align_gpu`, `:1815-1868`, itself called from
  `gpu_model_runner.py:1691-1704` right after
  `self.num_accepted_tokens.gpu[:num_reqs] = (output_token_ids != -1).sum(dim=1)`
  (`:1684`). Accepted lengths are device-side here, which is what the commit wants.
- `precopy_mamba_align_fused_kernel`, `mamba_utils.py:627-708`, whose
  `src_off = max(num_accepted - 1, 0)` (`:612`) becomes `0`, the commit having
  done the prefix work.
- the CPU-metadata path `preprocess_mamba`, `mamba_utils.py:1730-1753`, where
  `accept_token_bias = num_accepted_tokens_cpu[i] - 1` (`:1736`) becomes `0`.

**(c) Nothing else changes.** Specifically:
- `num_accepted_tokens` is already reset to 1 after a boundary copy
  (`mamba_utils.py:1743`, `:622-623`), which is exactly the post-commit
  requirement.
- The `src_col == dst_col` early return (`:684-685`, "kernels locate the initial
  state in-block via num_accepted") stays: within a block there is no commit,
  and the decode kernel's fused replay is what locates the state.
- CUDA graphs: no new buffer, so `GDNAttentionMetadataBuilder`'s graph-owned
  tensors (`vllm/v1/attention/backends/gdn_attn.py:226-282`) and
  `_B12xGdnDecodeStaging` (`qwen_gdn_linear_attn.py:93-149`) are unchanged, and
  the workspace lock (`gpu_model_runner.py:7223-7225`) is not a constraint.
- Prefix caching: speculative blocks are exclusively owned and unhashed
  (`single_type_kv_cache_manager.py:2280-2289`) and only the running/aligned
  block is cached (`:2188-2197`), so the record columns are never observed by
  another request.

**(d) The one invariant to confirm before enabling.** Columns
`0 .. num_spec` of the sliced block table must still hold what the previous step
wrote, or a record written at step N is gone at step N+1. The shipped kernel
already depends on this — it reads `state_indices[r, accepted - 1]` — so the
invariant is not new, but `_relocate_speculative_block`
(`single_type_kv_cache_manager.py:2280-2289`, called from `:2220-2228`) and
`remove_skipped_blocks` move and null spec blocks as the window advances, and
the commit at a boundary is the one place where the consequence changes from
"wrong checkpoint" to "record read as a state". Confirm it with an assertion in
`preprocess_mamba` before the first serving run.
