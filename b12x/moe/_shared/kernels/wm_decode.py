"""Weight-major fused NVFP4 MoE decode kernel (``wm`` backend) for SM120/SM121.

One launch per MoE call, one CTA per SM, and no inter-CTA communication other than
the output atomics: no grid barrier, no task queue, and no counters that need
resetting. See docs/moe-weight-major-decode.md for the bandwidth model.

Every CTA ranks the touched experts itself. It reads ``topk_ids`` (at most
``max_tokens * top_k`` routes) into shared memory, builds a touched-expert bitmap,
and orders the experts by ascending id. A work item is ``(expert, pass)``, where a
pass is one 16-row block of that expert's routed rows, taken in route order. A
second pass exists only when more than 16 rows pick the same expert. CTA ``b``
owns items ``b, b + grid, ...``. Every item streams the same bytes, which is its
expert's full w13 and down weights exactly once, so a round of items finishes
together.

Each item is one uniform sequence of ~11 KB weight stages through a
``g["stages"]``-slot cp.async ring (5 at I=320, 4 at I=640):

- FC1: for each 32-channel block, 32 up rows and then the same 32 gate rows. Each
  row group is swept over K in chunks of 640 (320 contiguous bytes per row) with the
  rows' K64 scale words gathered from the F8_128x4 plane.
- FC2: blocks of 64 (I=320) or 32 (I=640) ``down`` rows over the full intermediate K, stored as fully
  contiguous 10 KB spans plus 64 x (I/64) scale words.

Weight addresses depend only on the expert id, so the ring runs straight through the
FC1->FC2 turn and into the next item.

The numerics mirror the dynamic kernel's NVFP4 SiLU path step for step:

- the same per-16 activation quantizer and input scale;
- the same native ``m16n8k64 kind::mxf4nvf4`` block-scaled QMMA, with the fragment
  and scale mapping of ``nvfp4_phase1``, FP32 accumulation and ascending K;
- ``alpha * acc`` for gate and up, then ``SiLU(g) * u`` with
  ``rcp_approx(1 + exp(-g))`` under the same fast-math contract;
- BF16 rounding, then the same per-16 requantizer with the FC2 input scale;
- FC2 with ascending K inside an expert, then ``bf16(down_alpha * acc)``, multiplied
  by the router weight in FP32;
- the same BF16x2 atomic scatter-add into a pre-zeroed output.

The expert's FC2 partial is not split into 128-channel slices, so each output element
receives one atomic add per routed expert instead of one per (expert, slice).

Weight contract (the prepared NVFP4 storage that direct micro also consumes): the
w13 payload is ``[E][2I][K/2]`` in kernel order (up rows first, then gate), the
down payload is ``[E][K][I/2]``, and both scale planes are compact F8_128x4 E4M3
planes per expert.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, Uint8, Uint32, Uint64, dsl_user_op

from b12x._lib.intrinsics import (
    bfloat2_to_float2_scaled,
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    cp_async_u64_shared_global,
    fabs_f32,
    fmax_f32,
    get_ptr_as_int64,
    ld_global_v4_u32,
    ld_shared_bf16_to_f32,
    ld_shared_f32,
    ld_shared_u32,
    nvfp4_mma_m16n8k64_f32_e2m1,
    pack_f32x2_to_bfloat2,
    quantize_block_fp4,
    quantize_block_fp4_fast,
    scatter_add_bf16x2,
    shared_ptr_to_u32,
    st_shared_f32,
    st_shared_u32,
    st_shared_u8,
)
from b12x.moe._shared.kernels.wm_geometry import (
    FC1_ROWS,
    NUM_WARPS,
    PASS_ROWS,
    THREADS,
    wm_geometry,
)

QUANT_BATCH = 4


@dsl_user_op
def _popc(x: Uint32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(), [Uint32(x).ir_value(loc=loc, ip=ip)], "popc.b32 $0, $1;", "=r,r",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )


@dsl_user_op
def _ballot(pred: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(), [Int32(pred).ir_value(loc=loc, ip=ip)],
            "{.reg .pred p; setp.ne.s32 p, $1, 0; vote.sync.ballot.b32 $0, p, 0xffffffff;}",
            "=r,r", has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )


@dsl_user_op
def _lanemask_lt(*, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(), [], "mov.u32 $0, %lanemask_lt;", "=r",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )


@dsl_user_op
def _red_or_shared(addr: Int32, val: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None, [Int32(addr).ir_value(loc=loc, ip=ip), Uint32(val).ir_value(loc=loc, ip=ip)],
        "red.shared.or.b32 [$0], $1;", "r,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def _red_add_shared(addr: Int32, val: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None, [Int32(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "red.shared.add.s32 [$0], $1;", "r,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


class MoEWeightMajorDecodeKernel:
    """Weight-major fused NVFP4 SiLU MoE decode (see module docstring)."""

    def __init__(self, *, hidden_size: int, intermediate_size: int, num_experts: int,
                 top_k: int, max_tokens: int, fast_math: bool, ids_int64: bool,
                 probe: frozenset = frozenset()):
        self.g = wm_geometry(hidden_size=hidden_size, intermediate_size=intermediate_size,
                             num_experts=num_experts, top_k=top_k, max_tokens=max_tokens)
        self.fast_math = bool(fast_math)
        self.ids_int64 = bool(ids_int64)
        # Benchmark-only timing probes (wrong results): nocompute, noscale, noprologue.
        self.probe = frozenset(probe)
        self.shared_words = self.g["smem_bytes"] // 4

    @property
    def __cache_key__(self):
        g = self.g
        key = ("wm_decode", 1, g["K"], g["I"], g["E"], g["top_k"], g["max_tokens"],
               g["stages"], self.fast_math, self.ids_int64, tuple(sorted(self.probe)))
        # The default (64-row FC2) geometry keeps its original key.
        return key if g["fc2_rows"] == 64 else key + (("fc2_rows", g["fc2_rows"]),)

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        ids_ptr: cute.Pointer,
        tw_ptr: cute.Pointer,
        w13_ptr: cute.Pointer,
        sf13_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        sf2_ptr: cute.Pointer,
        in_gs_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        fc2_gs_ptr: cute.Pointer,
        dalpha_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        num_tokens: Int32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ):
        g = self.g
        E, K, I = g["E"], g["K"], g["I"]
        x = cute.make_tensor(x_ptr, cute.make_layout(Int32(g["max_tokens"] * K)))
        ids = cute.make_tensor(ids_ptr, cute.make_layout(Int32(g["routes"])))
        tw = cute.make_tensor(tw_ptr, cute.make_layout(Int32(g["routes"])))
        w13 = cute.make_tensor(w13_ptr, cute.make_layout(Int64(E * 2 * I * g["k2"])))
        sf13 = cute.make_tensor(sf13_ptr, cute.make_layout(Int64(E * g["sf13_expert"])))
        w2 = cute.make_tensor(w2_ptr, cute.make_layout(Int64(E * K * g["i2"])))
        sf2 = cute.make_tensor(sf2_ptr, cute.make_layout(Int64(E * g["sf2_expert"])))
        in_gs = cute.make_tensor(in_gs_ptr, cute.make_layout(E))
        alpha = cute.make_tensor(alpha_ptr, cute.make_layout(E))
        fc2_gs = cute.make_tensor(fc2_gs_ptr, cute.make_layout(E))
        dalpha = cute.make_tensor(dalpha_ptr, cute.make_layout(E))
        out = cute.make_tensor(out_ptr, cute.make_layout(Int32(g["max_tokens"] * K)))
        self.kernel(
            x, ids, tw, w13, sf13, w2, sf2, in_gs, alpha, fc2_gs, dalpha, out, num_tokens,
        ).launch(grid=(grid_x, Int32(1), Int32(1)), block=[THREADS, 1, 1], stream=stream)

    # ------------------------------------------------------------------ routing

    @cute.jit
    def _item_expert(self, base: Int32, item: Int32, d: Int32) -> Int32:
        """Expert id of work item ``item`` (pass 0 below ``d``, pass 1 above)."""
        g = self.g
        bm = base + Int32(g["off_bm"])
        pre = base + Int32(g["off_pre"])
        r = item
        if item >= d:
            bm = base + Int32(g["off_big"])
            pre = base + Int32(g["off_bpre"])
            r = item - d
        word = Int32(0)
        for w in cutlass.range_constexpr(1, g["bitmap_words"]):
            if Int32(ld_shared_u32(pre + Int32(4 * w))) <= r:
                word = Int32(w)
        bits = ld_shared_u32(bm + word * Int32(4))
        k = r - Int32(ld_shared_u32(pre + word * Int32(4)))
        while k > Int32(0):
            bits = bits & (bits - Uint32(1))
            k -= Int32(1)
        low = bits & (Uint32(0) - bits)
        return word * Int32(32) + _popc(low - Uint32(1))

    @cute.jit
    def _gather_pass(self, ids: cute.Tensor, tw: cute.Tensor, base: Int32, tid: Int32,
                     expert: Int32, pass_idx: Int32, routes: Int32):
        """Warp 0 collects the pass's routed rows in route order (deterministic)."""
        g = self.g
        if tid < Int32(32):
            lane = tid
            below = _lanemask_lt()
            seen = Int32(0)
            first = pass_idx * Int32(PASS_ROWS)
            # Issue every route load before the first ballot: one latency, not one per chunk.
            route_e = cute.make_rmem_tensor((g["route_chunks"],), cutlass.Int32)
            for chunk in cutlass.range_constexpr(g["route_chunks"]):
                p = Int32(chunk * 32) + lane
                route_e[chunk] = Int32(-1)
                if p < routes:
                    route_e[chunk] = Int32(ids[p])
            for chunk in cutlass.range_constexpr(g["route_chunks"]):
                p = Int32(chunk * 32) + lane
                hit = Int32(0)
                if route_e[chunk] == expert:
                    hit = Int32(1)
                mask = _ballot(hit)
                slot = seen + _popc(mask & below) - first
                if hit != Int32(0):
                    if slot >= Int32(0):
                        if slot < Int32(PASS_ROWS):
                            st_shared_u32(base + Int32(g["off_tok"]) + slot * Int32(4),
                                          Uint32(p // Int32(g["top_k"])))
                            st_shared_f32(base + Int32(g["off_wgt"]) + slot * Int32(4),
                                          tw[p].to(cutlass.Float32))
                            if cutlass.const_expr("off_route" in g):
                                st_shared_u32(base + Int32(g["off_route"]) + slot * Int32(4), Uint32(p))
                seen += _popc(mask)
            if lane == Int32(0):
                rows = seen - first
                if rows > Int32(PASS_ROWS):
                    rows = Int32(PASS_ROWS)
                if rows < Int32(0):
                    rows = Int32(0)
                st_shared_u32(base + Int32(g["off_nrows"]), Uint32(rows))

    @cute.jit
    def _quantize_rows(self, x: cute.Tensor, in_gs: cute.Tensor, base: Int32, tid: Int32,
                       expert: Int32, n_rows: Int32):
        """Quantize the pass rows to NVFP4 into the A tile (zero rows past n_rows).

        Thread ``tid`` owns blocks ``tid + THREADS * i``; each batch of QUANT_BATCH blocks
        issues all of its global loads before quantizing, so the per-item prologue pays
        a handful of load latencies instead of one per block.
        """
        g = self.g
        K = g["K"]
        blocks = K // 16
        per_thread = PASS_ROWS * blocks // THREADS
        batch = QUANT_BATCH
        gs = in_gs[expert].to(cutlass.Float32)
        a_base = base + Int32(g["off_a"])
        as_base = base + Int32(g["off_as"])
        for grp in cutlass.range_constexpr(per_thread // batch):
            raw = cute.make_rmem_tensor((batch * 8,), cutlass.Uint32)
            for u in cutlass.range_constexpr(batch):
                idx = tid + Int32((grp * batch + u) * THREADS)
                r = idx // Int32(blocks)
                b = idx - r * Int32(blocks)
                for w in cutlass.range_constexpr(8):
                    raw[u * 8 + w] = Uint32(0)
                if r < n_rows:
                    tok = Int32(ld_shared_u32(base + Int32(g["off_tok"]) + r * Int32(4)))
                    src = get_ptr_as_int64(x, tok * Int32(K) + b * Int32(16))
                    for h in cutlass.range_constexpr(2):
                        v0, v1, v2, v3 = ld_global_v4_u32(src + Int64(16 * h))
                        raw[u * 8 + 4 * h] = v0
                        raw[u * 8 + 4 * h + 1] = v1
                        raw[u * 8 + 4 * h + 2] = v2
                        raw[u * 8 + 4 * h + 3] = v3
            for u in cutlass.range_constexpr(batch):
                idx = tid + Int32((grp * batch + u) * THREADS)
                r = idx // Int32(blocks)
                b = idx - r * Int32(blocks)
                values = cute.make_rmem_tensor((16,), cutlass.Float32)
                block_max = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(8):
                    lo, hi = bfloat2_to_float2_scaled(raw[u * 8 + w], cutlass.Float32(1.0))
                    values[2 * w] = lo
                    values[2 * w + 1] = hi
                for e in cutlass.range_constexpr(16):
                    block_max = fmax_f32(block_max, fabs_f32(values[e]))
                packed = Uint64(0)
                scale = Uint8(0)
                if cutlass.const_expr(self.fast_math):
                    packed, scale = quantize_block_fp4_fast(values, block_max, gs)
                else:
                    packed, scale = quantize_block_fp4(values, block_max, gs)
                # Rows past n_rows are all-zero blocks, which quantize to zero payload
                # and a zero scale byte.
                dst = a_base + r * Int32(g["a_stride"]) + b * Int32(8)
                st_shared_u32(dst, Uint32(packed & Uint64(0xFFFFFFFF)))
                st_shared_u32(dst + Int32(4), Uint32(packed >> Uint64(32)))
                st_shared_u8(as_base + r * Int32(4 * g["as_words"]) + b, scale)

    @cute.jit
    def _quantize_intermediate(self, fc2_gs: cute.Tensor, base: Int32, tid: Int32,
                               expert: Int32):
        """BF16 SiLU*up tile -> NVFP4 FC2 A tile (aliases the dead FC1 A tile)."""
        g = self.g
        I = g["I"]
        blocks = I // 16
        gs = fc2_gs[expert].to(cutlass.Float32)
        h_base = base + Int32(g["off_h"])
        hq_base = base + Int32(g["off_a"])
        hs_base = base + Int32(g["off_as"])
        idx = tid
        while idx < Int32(PASS_ROWS * blocks):
            r = idx // Int32(blocks)
            b = idx - r * Int32(blocks)
            values = cute.make_rmem_tensor((16,), cutlass.Float32)
            block_max = cutlass.Float32(0.0)
            for e in cutlass.range_constexpr(16):
                v = ld_shared_bf16_to_f32(h_base + (r * Int32(I) + b * Int32(16) + Int32(e)) * Int32(2))
                values[e] = v
                block_max = fmax_f32(block_max, fabs_f32(v))
            packed = Uint64(0)
            scale = Uint8(0)
            if cutlass.const_expr(self.fast_math):
                packed, scale = quantize_block_fp4_fast(values, block_max, gs)
            else:
                packed, scale = quantize_block_fp4(values, block_max, gs)
            dst = hq_base + r * Int32(g["hq_stride"]) + b * Int32(8)
            st_shared_u32(dst, Uint32(packed & Uint64(0xFFFFFFFF)))
            st_shared_u32(dst + Int32(4), Uint32(packed >> Uint64(32)))
            st_shared_u8(hs_base + r * Int32(g["hs_stride"]) + b, scale)
            idx += Int32(THREADS)

    # ------------------------------------------------------------------ streaming

    @cute.jit
    def _fc1_channel(self, cb: Int32, n: Int32) -> Int32:
        """Intermediate channel of local FC1 row n (0..7) in channel block cb.

        Block cb covers channels ch0..ch0+3 and ch0+32..ch0+35 (ch0 = 64 * (cb // 8) +
        4 * (cb % 8)), so each row's F8_128x4 scale word shares an 8 B pair with the
        word of the row 32 channels up. Needs I % 64 == 0 (wm_geometry checks it).
        """
        return ((cb >> Int32(3)) * Int32(64) + (cb & Int32(7)) * Int32(4)
                + (n & Int32(3)) + (n >> Int32(2)) * Int32(32))

    @cute.jit
    def _issue_stage(self, w13: cute.Tensor, sf13: cute.Tensor, w2: cute.Tensor,
                     sf2: cute.Tensor, base: Int32, tid: Int32, expert: Int32,
                     stage: Int32, slot: Int32):
        g = self.g
        K, I = g["K"], g["I"]
        slot_base = base + Int32(g["off_ring"]) + slot * Int32(g["slot_bytes"])
        sf_base = slot_base + Int32(g["slot_payload"])
        if stage < Int32(g["fc1_stages"]):
            cb = stage // Int32(2 * g["kchunks"])
            half = (stage // Int32(g["kchunks"])) % Int32(2)
            kc = stage % Int32(g["kchunks"])
            row0 = half * Int32(I)
            per_row = g["k2"] // 16
            # Local rows 0..3 and 4..7 are two contiguous spans of 4 * K/2 bytes
            # (channels ch0..ch0+3 and ch0+32..ch0+35, see _fc1_channel).
            for i in cutlass.range_constexpr(g["fc1_chunks"] // THREADS):
                idx = tid + Int32(i * THREADS)
                r = idx // Int32(per_row)
                v = idx - r * Int32(per_row)
                row = row0 + self._fc1_channel(cb, r)
                src = (Int64(expert) * Int64(2 * I) + Int64(row)) * Int64(g["k2"]) + Int64(v * Int32(16))
                cp_async4_shared_global(slot_base + r * Int32(g["p1"]) + v * Int32(16),
                                        get_ptr_as_int64(w13, src))
            e_base = Int64(expert) * Int64(g["sf13_expert"])
            # Rows x and x + 32 own adjacent words of one F8_128x4 16 B chunk: one 8 B
            # copy per (row pair, K64). Shared layout [K64][pair 0..3][row, row + 32].
            for i in cutlass.range_constexpr(0 if "noscale" in self.probe else -(-(g["fc1_words"] // 2) // THREADS)):
                idx = tid + Int32(i * THREADS)
                if idx < Int32(g["fc1_words"] // 2):
                    k64 = kc * Int32(g["fc1_k64"]) + (idx >> Int32(2))
                    row = row0 + self._fc1_channel(cb, idx & Int32(3))
                    off = ((row >> Int32(7)) * Int32(g["sf13_atom"]) + k64 * Int32(512)
                           + (row & Int32(31)) * Int32(16) + ((row & Int32(127)) >> Int32(5)) * Int32(4))
                    cp_async_u64_shared_global(sf_base + idx * Int32(8),
                                               get_ptr_as_int64(sf13, e_base + Int64(off)))
        else:
            rb = stage - Int32(g["fc1_stages"])
            row0 = rb * Int32(g["fc2_rows"])
            per_row = g["i2"] // 16
            span = (Int64(expert) * Int64(K) + Int64(row0)) * Int64(g["i2"])
            for i in cutlass.range_constexpr(-(-g["fc2_chunks"] // THREADS)):
                idx = tid + Int32(i * THREADS)
                if idx < Int32(g["fc2_chunks"]):
                    r = idx // Int32(per_row)
                    v = idx - r * Int32(per_row)
                    cp_async4_shared_global(slot_base + r * Int32(g["p2"]) + v * Int32(16),
                                            get_ptr_as_int64(w2, span + Int64(idx * Int32(16))))
            e_base = Int64(expert) * Int64(g["sf2_expert"])
            if cutlass.const_expr(g["fc2_rows"] == 32 and "noscale" not in self.probe):
                # 32-row stages own one word of each F8_128x4 16 B chunk: 4 B per
                # (row, K64); shared layout [K64][r 0..31].
                for i in cutlass.range_constexpr(-(-g["fc2_words"] // THREADS)):
                    idx = tid + Int32(i * THREADS)
                    if idx < Int32(g["fc2_words"]):
                        j = idx >> Int32(5)
                        row = row0 + (idx & Int32(31))
                        off = ((row >> Int32(7)) * Int32(g["sf2_atom"]) + j * Int32(512)
                               + (row & Int32(31)) * Int32(16) + ((row & Int32(127)) >> Int32(5)) * Int32(4))
                        cp_async_u32_shared_global(sf_base + idx * Int32(4),
                                                   get_ptr_as_int64(sf2, e_base + Int64(off)))
            # 8 B per (row pair r, r + 32; K64); shared layout [K64][r 0..31][row, row + 32].
            for i in cutlass.range_constexpr(0 if ("noscale" in self.probe or g["fc2_rows"] != 64) else -(-(g["fc2_words"] // 2) // THREADS)):
                idx = tid + Int32(i * THREADS)
                if idx < Int32(g["fc2_words"] // 2):
                    j = idx >> Int32(5)
                    row = row0 + (idx & Int32(31))
                    off = ((row >> Int32(7)) * Int32(g["sf2_atom"]) + j * Int32(512)
                           + (row & Int32(31)) * Int32(16) + ((row & Int32(127)) >> Int32(5)) * Int32(4))
                    cp_async_u64_shared_global(sf_base + idx * Int32(8),
                                               get_ptr_as_int64(sf2, e_base + Int64(off)))

    @cute.jit
    def _route_prologue(self, ids: cute.Tensor, base: Int32, tid: Int32, num_tokens: Int32):
        """Routing, redundantly in every CTA: touched-expert bitmaps and their prefix counts.

        Returns (d, d2, routes): experts with >= 1 row, experts with > PASS_ROWS rows,
        and the number of live routes.
        """
        g = self.g
        E = g["E"]
        lane = tid & Int32(31)
        nw = g["bitmap_words"]
        for i in cutlass.range_constexpr(-(-nw // THREADS)):
            w = tid + Int32(i * THREADS)
            if w < Int32(nw):
                st_shared_u32(base + Int32(g["off_bm"]) + w * Int32(4), Uint32(0))
                st_shared_u32(base + Int32(g["off_big"]) + w * Int32(4), Uint32(0))
        if cutlass.const_expr(g["passes"] > 1):
            for i in cutlass.range_constexpr(-(-E // THREADS)):
                ez = tid + Int32(i * THREADS)
                if ez < Int32(E):
                    st_shared_u32(base + Int32(g["off_cnt"]) + ez * Int32(4), Uint32(0))
        cute.arch.sync_threads()
        routes = num_tokens * Int32(g["top_k"])
        for i in cutlass.range_constexpr(-(-g["routes"] // THREADS)):
            p = tid + Int32(i * THREADS)
            route_e = Int32(-1)
            if p < routes:
                route_e = Int32(ids[p])
            if route_e >= Int32(0):
                if route_e < Int32(E):
                    _red_or_shared(base + Int32(g["off_bm"]) + (route_e >> Int32(5)) * Int32(4),
                                   Uint32(1) << Uint32(route_e & Int32(31)))
                    if cutlass.const_expr(g["passes"] > 1):
                        _red_add_shared(base + Int32(g["off_cnt"]) + route_e * Int32(4), Int32(1))
        cute.arch.sync_threads()
        if cutlass.const_expr(g["passes"] > 1):
            for i in cutlass.range_constexpr(-(-E // THREADS)):
                eb = tid + Int32(i * THREADS)
                if eb < Int32(E):
                    if Int32(ld_shared_u32(base + Int32(g["off_cnt"]) + eb * Int32(4))) > Int32(PASS_ROWS):
                        _red_or_shared(base + Int32(g["off_big"]) + (eb >> Int32(5)) * Int32(4),
                                       Uint32(1) << Uint32(eb & Int32(31)))
            cute.arch.sync_threads()
        # Exclusive prefix popcounts of both bitmaps (warp 0; nw <= 32).
        if tid < Int32(32):
            for which in cutlass.range_constexpr(2):
                bm_off = g["off_bm"] if which == 0 else g["off_big"]
                pre_off = g["off_pre"] if which == 0 else g["off_bpre"]
                cnt = Int32(0)
                if lane < Int32(nw):
                    cnt = _popc(ld_shared_u32(base + Int32(bm_off) + lane * Int32(4)))
                incl = cnt
                for sh in cutlass.range_constexpr(5):
                    other = cute.arch.shuffle_sync_up(incl, offset=1 << sh)
                    if lane >= Int32(1 << sh):
                        incl = incl + other
                if lane < Int32(nw):
                    st_shared_u32(base + Int32(pre_off) + lane * Int32(4), Uint32(incl - cnt))
                if lane == Int32(nw - 1):
                    st_shared_u32(base + Int32(pre_off) + Int32(4 * nw), Uint32(incl))
        cute.arch.sync_threads()
        d = Int32(ld_shared_u32(base + Int32(g["off_pre"] + 4 * nw)))
        d2 = Int32(ld_shared_u32(base + Int32(g["off_bpre"] + 4 * nw)))
        return d, d2, routes

    # ------------------------------------------------------------------ stage compute

    @cute.jit
    def _store_h(self, hbuf: cute.Tensor, base: Int32, h_base: Int32, row: Int32, ch: Int32,
                 packed: Uint32):
        """One BF16x2 of the SiLU*up tile: shared memory (wm) or the global h rows (flint)."""
        st_shared_u32(h_base + (row * Int32(self.g["I"]) + ch) * Int32(2), packed)

    @cute.jit
    def _fc1_step(self, alpha: cute.Tensor, hbuf: cute.Tensor, base: Int32, tid: Int32,
                  warp: Int32, q: Int32, c: Int32, s: Int32, cur_e: Int32,
                  slot_base: Int32, sf_base: Int32):
        """FC1 stage s (channel block s // 2, up or gate half) of the current item."""
        g = self.g
        I = g["I"]
        a_base = base + Int32(g["off_a"])
        as_base = base + Int32(g["off_as"])
        h_base = base + Int32(g["off_h"])
        cb = s // Int32(2)
        half = s % Int32(2)
        # Split-K: warp w contracts K64 slices [w * warp_k64, (w + 1) * warp_k64)
        # of the stage's FC1_ROWS rows, then publishes its FP32 partial.
        acc = cute.make_rmem_tensor((4,), cutlass.Float32)
        acc.fill(0.0)
        sf_row = q + Int32(8) * (c & Int32(1))
        for jj in cutlass.range_constexpr(g["warp_k64"]):
            k64 = warp * Int32(g["warp_k64"]) + Int32(jj)
            a_off = k64 * Int32(32) + Int32(4) * c
            a0 = ld_shared_u32(a_base + q * Int32(g["a_stride"]) + a_off)
            a1 = ld_shared_u32(a_base + (q + Int32(8)) * Int32(g["a_stride"]) + a_off)
            a2 = ld_shared_u32(a_base + q * Int32(g["a_stride"]) + a_off + Int32(16))
            a3 = ld_shared_u32(a_base + (q + Int32(8)) * Int32(g["a_stride"]) + a_off + Int32(16))
            sfa = ld_shared_u32(as_base + sf_row * Int32(4 * g["as_words"]) + k64 * Int32(4))
            b_off = slot_base + q * Int32(g["p1"]) + k64 * Int32(32) + Int32(4) * c
            b0 = ld_shared_u32(b_off)
            b1 = ld_shared_u32(b_off + Int32(16))
            sfb = ld_shared_u32(sf_base + ((k64 * Int32(4) + (q & Int32(3))) * Int32(2)
                                           + (q >> Int32(2))) * Int32(4))
            d0, d1, d2_, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
                acc[0], acc[1], acc[2], acc[3], a0, a1, a2, a3, b0, b1, sfa, sfb)
            acc[0] = d0
            acc[1] = d1
            acc[2] = d2_
            acc[3] = d3
        # part[half][warp][token 0..15][channel 0..7], FP32.
        part = base + Int32(g["off_part"]) + (half * Int32(NUM_WARPS) + warp) * Int32(PASS_ROWS * FC1_ROWS * 4)
        st_shared_f32(part + (q * Int32(FC1_ROWS) + Int32(2) * c) * Int32(4), acc[0])
        st_shared_f32(part + (q * Int32(FC1_ROWS) + Int32(2) * c + Int32(1)) * Int32(4), acc[1])
        st_shared_f32(part + ((q + Int32(8)) * Int32(FC1_ROWS) + Int32(2) * c) * Int32(4), acc[2])
        st_shared_f32(part + ((q + Int32(8)) * Int32(FC1_ROWS) + Int32(2) * c + Int32(1)) * Int32(4), acc[3])
        if half == Int32(1):
            cute.arch.sync_threads()
            if tid < Int32(PASS_ROWS * FC1_ROWS // 2):
                row = tid >> Int32(2)
                col2 = (tid & Int32(3)) * Int32(2)
                al = alpha[cur_e].to(cutlass.Float32)
                pbase = base + Int32(g["off_part"])
                acts = cute.make_rmem_tensor((2,), cutlass.Float32)
                for e2 in cutlass.range_constexpr(2):
                    elem = (row * Int32(FC1_ROWS) + col2 + Int32(e2)) * Int32(4)
                    usum = ld_shared_f32(pbase + elem)
                    gsum = ld_shared_f32(pbase + Int32(NUM_WARPS * PASS_ROWS * FC1_ROWS * 4) + elem)
                    for w in cutlass.range_constexpr(1, NUM_WARPS):
                        usum = usum + ld_shared_f32(pbase + Int32(w * PASS_ROWS * FC1_ROWS * 4) + elem)
                        gsum = gsum + ld_shared_f32(
                            pbase + Int32((NUM_WARPS + w) * PASS_ROWS * FC1_ROWS * 4) + elem)
                    gv = al * gsum
                    uv = al * usum
                    sig = cute.arch.rcp_approx(
                        cutlass.Float32(1.0) + cute.math.exp(-gv, fastmath=self.fast_math))
                    acts[e2] = gv * sig * uv
                self._store_h(hbuf, base, h_base, row, self._fc1_channel(cb, col2),
                              pack_f32x2_to_bfloat2(acts[0], acts[1]))

    @cute.jit
    def _fc2_step(self, dalpha: cute.Tensor, out: cute.Tensor, base: Int32, warp: Int32,
                  q: Int32, c: Int32, s: Int32, cur_e: Int32, n_rows: Int32,
                  slot_base: Int32, sf_base: Int32):
        """FC2 stage s (down rows block s - fc1_stages) of the current item."""
        g = self.g
        K = g["K"]
        a_base = base + Int32(g["off_a"])
        as_base = base + Int32(g["off_as"])
        out_row = Int32(K)
        rb = s - Int32(g["fc1_stages"])
        hq_base = a_base
        hs_base = as_base
        sf_row = q + Int32(8) * (c & Int32(1))
        dal = dalpha[cur_e].to(cutlass.Float32)
        tiles = g["fc2_rows"] // (8 * NUM_WARPS)
        for t in cutlass.range_constexpr(tiles):
            nt = warp * Int32(tiles) + Int32(t)
            b_row = nt * Int32(8) + q
            acc = cute.make_rmem_tensor((4,), cutlass.Float32)
            acc.fill(0.0)
            for jj in cutlass.range_constexpr(g["fc2_k64"]):
                a_off = Int32(jj * 32) + Int32(4) * c
                a0 = ld_shared_u32(hq_base + q * Int32(g["hq_stride"]) + a_off)
                a1 = ld_shared_u32(hq_base + (q + Int32(8)) * Int32(g["hq_stride"]) + a_off)
                a2 = ld_shared_u32(hq_base + q * Int32(g["hq_stride"]) + a_off + Int32(16))
                a3 = ld_shared_u32(hq_base + (q + Int32(8)) * Int32(g["hq_stride"]) + a_off + Int32(16))
                sfa = ld_shared_u32(hs_base + sf_row * Int32(g["hs_stride"]) + Int32(jj * 4))
                b_off = slot_base + b_row * Int32(g["p2"]) + Int32(jj * 32) + Int32(4) * c
                b0 = ld_shared_u32(b_off)
                b1 = ld_shared_u32(b_off + Int32(16))
                if cutlass.const_expr(g["fc2_rows"] == 64):
                    sfb = ld_shared_u32(sf_base + ((Int32(jj * 32) + (b_row & Int32(31))) * Int32(2)
                                                   + (b_row >> Int32(5))) * Int32(4))
                else:
                    sfb = ld_shared_u32(sf_base + (Int32(jj * 32) + b_row) * Int32(4))
                d0, d1, d2_, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
                    acc[0], acc[1], acc[2], acc[3], a0, a1, a2, a3, b0, b1, sfa, sfb)
                acc[0] = d0
                acc[1] = d1
                acc[2] = d2_
                acc[3] = d3
            col = rb * Int32(g["fc2_rows"]) + nt * Int32(8) + Int32(2) * c
            for hr in cutlass.range_constexpr(2):
                row = q + Int32(8 * hr)
                if row < n_rows:
                    tok = Int32(ld_shared_u32(base + Int32(g["off_tok"]) + row * Int32(4)))
                    wv = ld_shared_f32(base + Int32(g["off_wgt"]) + row * Int32(4))
                    y0, y1 = bfloat2_to_float2_scaled(
                        pack_f32x2_to_bfloat2(dal * acc[2 * hr], dal * acc[2 * hr + 1]), wv)
                    scatter_add_bf16x2(get_ptr_as_int64(out, tok * out_row + col), y0, y1)

    # ------------------------------------------------------------------ kernel

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        ids: cute.Tensor,
        tw: cute.Tensor,
        w13: cute.Tensor,
        sf13: cute.Tensor,
        w2: cute.Tensor,
        sf2: cute.Tensor,
        in_gs: cute.Tensor,
        alpha: cute.Tensor,
        fc2_gs: cute.Tensor,
        dalpha: cute.Tensor,
        out: cute.Tensor,
        num_tokens: Int32,
    ):
        g = self.g
        E, K, I = g["E"], g["K"], g["I"]
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        bid = Int32(bidx)
        grid = Int32(gdim)
        warp = tid >> Int32(5)
        lane = tid & Int32(31)
        q = lane >> Int32(2)
        c = lane & Int32(3)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words], 1024
            ]

        storage = smem.allocate(Storage)
        base = shared_ptr_to_u32(storage.words.data_ptr())

        d, d2, routes = self._route_prologue(ids, base, tid, num_tokens)
        items = d + d2
        mine = Int32(0)
        if bid < items:
            mine = (items - bid + grid - Int32(1)) // grid
        spi = Int32(g["spi"])
        total = mine * spi

        # ---- the per-CTA stage stream
        iss_e = Int32(0)
        if mine > Int32(0):
            iss_e = self._item_expert(base, bid, d)
        stages = g["stages"]
        for s0 in cutlass.range_constexpr(stages - 1):
            if Int32(s0) < total:
                self._issue_stage(w13, sf13, w2, sf2, base, tid, iss_e, Int32(s0), Int32(s0))
            cute.arch.cp_async_commit_group()

        cur_e = Int32(0)
        n_rows = Int32(0)

        step = Int32(0)
        while step < total:
            j = step // spi
            s = step - j * spi
            if s == Int32(0):
                item = bid + j * grid
                cur_e = self._item_expert(base, item, d)
                pass_idx = Int32(0)
                if item >= d:
                    pass_idx = Int32(1)
                if cutlass.const_expr("noprologue" not in self.probe):
                    self._gather_pass(ids, tw, base, tid, cur_e, pass_idx, routes)
                    cute.arch.sync_threads()
                    n_rows = Int32(ld_shared_u32(base + Int32(g["off_nrows"])))
                    self._quantize_rows(x, in_gs, base, tid, cur_e, n_rows)

            # Prefetch stage step + stages - 1 into the slot freed last iteration.
            nxt = step + Int32(stages - 1)
            if nxt < total:
                nj = nxt // spi
                ns = nxt - nj * spi
                if ns == Int32(0):
                    iss_e = self._item_expert(base, bid + nj * grid, d)
                self._issue_stage(w13, sf13, w2, sf2, base, tid, iss_e, ns, nxt % Int32(stages))
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(stages - 1)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()

            slot_base = base + Int32(g["off_ring"]) + (step % Int32(stages)) * Int32(g["slot_bytes"])
            sf_base = slot_base + Int32(g["slot_payload"])
            if cutlass.const_expr("nocompute" not in self.probe):
                if s < Int32(g["fc1_stages"]):
                    self._fc1_step(alpha, out, base, tid, warp, q, c, s, cur_e, slot_base, sf_base)
                    if s == Int32(g["fc1_stages"] - 1):
                        cute.arch.sync_threads()
                        self._quantize_intermediate(fc2_gs, base, tid, cur_e)
                else:
                    self._fc2_step(dalpha, out, base, warp, q, c, s, cur_e, n_rows, slot_base, sf_base)
            cute.arch.sync_threads()
            step += Int32(1)

        cute.arch.cp_async_wait_group(0)
