"""flint: byte-balanced two-phase NVFP4 MoE decode kernel for SM120/SM121.

Same stage stream, QMMA and numerics as the weight-major ``wm`` kernel
(``wm_decode.py``), with a different work split. ``wm`` gives each CTA whole
experts, so a call with D touched experts runs ceil(D / grid) rounds and the last
round leaves SMs idle (D=60 on 48 SMs: 12 CTAs stream the second round alone).
flint splits every call into equal byte ranges over all CTAs:

- Phase A (FC1). The unit is (item, 8-channel block): 8 up rows and the same 8
  gate rows over full K, i.e. two ``wm`` FC1 stages. The ``items x I/8`` units are
  cut into ``grid`` contiguous ranges, so CTA ranges differ by at most one unit
  (~22 KB). Each unit ends with SiLU(alpha g) x (alpha u) for its 8 channels,
  rounded to BF16 and stored to a global h row per route (``hbuf[route][I]``).
- One grid barrier (every CTA is resident: 1 CTA/SM, grid <= SMs).
- Phase B (FC2). The unit is (item, ``fc2_rows`` down rows) over full I, one
  ``wm`` FC2 stage. The ``items x K/fc2_rows`` units are cut the same way. On an
  item change a CTA reloads the item's h rows from L2, requantizes them with the
  ``wm`` per-16 quantizer and runs the ``wm`` FC2 stage code, ending in the same
  BF16x2 atomic scatter-add (one add per routed expert and output element).

Weight addresses depend only on routing, so the cp.async ring keeps prefetching the
first FC2 stages while the CTA waits at the barrier.

Numerics equal ``wm`` exactly up to the order of the output atomics: the same
quantizers and scales, the same QMMA fragments and K order, the same FP32 split-K
reduction order, SiLU and BF16 rounding of h (the only change is that h goes
through global memory instead of shared memory), and the same FC2 epilogue.

Scratch: ``hbuf`` needs ``max_tokens * top_k * I * 2`` bytes. The launcher uses the
dynamic workspace's ``packed_input`` (``rows_padded * K/2`` bytes, rows_padded >=
routed rows, so it fits whenever ``2 I <= K/2``). The barrier uses slot 0 of the
workspace's ``barrier_count`` (left at zero after every launch) and
``barrier_epoch`` (incremented once per launch).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    atomic_add_global_i32,
    get_ptr_as_int64,
    ld_global_acquire_i32,
    ld_global_cg_v4_u32,
    ld_shared_u32,
    shared_ptr_to_u32,
    spin_wait_global_eq_i32,
    st_global_i32,
    st_global_release_i32,
    st_global_u32,
    st_shared_u32,
    threadfence,
)
from b12x.moe._shared.kernels.wm_decode import MoEWeightMajorDecodeKernel
from b12x.moe._shared.kernels.wm_geometry import THREADS, flint_geometry


class MoEFlintDecodeKernel(MoEWeightMajorDecodeKernel):
    """Byte-balanced two-phase NVFP4 SiLU MoE decode (see module docstring)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.g = flint_geometry(self.g)
        self.shared_words = self.g["smem_bytes"] // 4

    @property
    def __cache_key__(self):
        return ("flint_decode", 1) + super().__cache_key__[1:]

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
        h_ptr: cute.Pointer,
        cnt_ptr: cute.Pointer,
        epoch_ptr: cute.Pointer,
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
        hbuf = cute.make_tensor(h_ptr, cute.make_layout(Int32(g["routes"] * I * 2)))
        cnt = cute.make_tensor(cnt_ptr, cute.make_layout(1))
        epoch = cute.make_tensor(epoch_ptr, cute.make_layout(1))
        self.kernel(
            x, ids, tw, w13, sf13, w2, sf2, in_gs, alpha, fc2_gs, dalpha, out, hbuf, cnt, epoch,
            num_tokens,
        ).launch(grid=(grid_x, Int32(1), Int32(1)), block=[THREADS, 1, 1], stream=stream)

    # ------------------------------------------------------------------ h rows

    @cute.jit
    def _store_h(self, hbuf: cute.Tensor, base: Int32, h_base: Int32, row: Int32, ch: Int32,
                 packed: Uint32):
        """BF16x2 of row ``row`` -> the global h row of that row's route."""
        g = self.g
        if row < Int32(ld_shared_u32(base + Int32(g["off_nrows"]))):
            p = Int32(ld_shared_u32(base + Int32(g["off_route"]) + row * Int32(4)))
            st_global_u32(get_ptr_as_int64(hbuf, (p * Int32(g["I"]) + ch) * Int32(2)), packed)

    @cute.jit
    def _load_h(self, hbuf: cute.Tensor, base: Int32, tid: Int32, n_rows: Int32):
        """The item's h rows (route order) -> the shared h tile; rows past n_rows are zero."""
        g = self.g
        row_vec = g["I"] * 2 // 16
        h_base = base + Int32(g["off_h"])
        for i in cutlass.range_constexpr(-(-g["h_vec"] // THREADS)):
            idx = tid + Int32(i * THREADS)
            if idx < Int32(g["h_vec"]):
                r = idx // Int32(row_vec)
                v = idx - r * Int32(row_vec)
                w0 = Uint32(0)
                w1 = Uint32(0)
                w2 = Uint32(0)
                w3 = Uint32(0)
                if r < n_rows:
                    p = Int32(ld_shared_u32(base + Int32(g["off_route"]) + r * Int32(4)))
                    # L2-only load: the row was written by another CTA of this launch.
                    w0, w1, w2, w3 = ld_global_cg_v4_u32(
                        get_ptr_as_int64(hbuf, p * Int32(g["I"] * 2) + v * Int32(16)))
                dst = h_base + idx * Int32(16)
                st_shared_u32(dst, w0)
                st_shared_u32(dst + Int32(4), w1)
                st_shared_u32(dst + Int32(8), w2)
                st_shared_u32(dst + Int32(12), w3)

    @cute.jit
    def _grid_barrier(self, cnt: cute.Tensor, epoch: cute.Tensor, tid: Int32, grid: Int32):
        """Sense-reversing grid barrier; leaves the counter at zero for the next launch."""
        cute.arch.sync_threads()
        if tid == Int32(0):
            threadfence()
            cnt_addr = get_ptr_as_int64(cnt, Int32(0))
            epoch_addr = get_ptr_as_int64(epoch, Int32(0))
            e = ld_global_acquire_i32(epoch_addr)
            old = atomic_add_global_i32(cnt_addr, Int32(1))
            if old == grid - Int32(1):
                st_global_i32(cnt_addr, Int32(0))
                threadfence()
                st_global_release_i32(epoch_addr, e + Int32(1))
            else:
                spin_wait_global_eq_i32(epoch_addr, e)
        cute.arch.sync_threads()

    # ------------------------------------------------------------------ schedule

    @cute.jit
    def _map_a(self, a0: Int32, i: Int32):
        """Phase-A local step -> (item, stage within the item)."""
        g = self.g
        gs = Int32(2) * a0 + i
        item = gs // Int32(g["fc1_stages"])
        return item, gs - item * Int32(g["fc1_stages"])

    @cute.jit
    def _map_b(self, b0: Int32, i: Int32):
        """Phase-B local step -> (item, stage within the item)."""
        g = self.g
        u = b0 + i
        item = u // Int32(g["fc2_units"])
        return item, Int32(g["fc1_stages"]) + (u - item * Int32(g["fc2_units"]))

    @cute.jit
    def _issue_step(self, w13: cute.Tensor, sf13: cute.Tensor, w2: cute.Tensor, sf2: cute.Tensor,
                    base: Int32, tid: Int32, d: Int32, a0: Int32, b0: Int32, n_a: Int32,
                    step: Int32, slot: Int32):
        item = Int32(0)
        ws = Int32(0)
        if step < n_a:
            item, ws = self._map_a(a0, step)
        else:
            item, ws = self._map_b(b0, step - n_a)
        self._issue_stage(w13, sf13, w2, sf2, base, tid, self._item_expert(base, item, d), ws, slot)

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
        hbuf: cute.Tensor,
        cnt: cute.Tensor,
        epoch: cute.Tensor,
        num_tokens: Int32,
    ):
        g = self.g
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
        ua = items * Int32(g["fc1_units"])
        ub = items * Int32(g["fc2_units"])
        a0 = bid * ua // grid
        b0 = bid * ub // grid
        n_a = Int32(2) * ((bid + Int32(1)) * ua // grid - a0)
        n_b = (bid + Int32(1)) * ub // grid - b0
        total = n_a + n_b

        stages = g["stages"]
        for s0 in cutlass.range_constexpr(stages - 1):
            if Int32(s0) < total:
                self._issue_step(w13, sf13, w2, sf2, base, tid, d, a0, b0, n_a, Int32(s0), Int32(s0))
            cute.arch.cp_async_commit_group()

        cur_item = Int32(-1)
        cur_e = Int32(0)
        n_rows = Int32(0)
        step = Int32(0)
        while step < total:
            item = Int32(0)
            ws = Int32(0)
            if step < n_a:
                item, ws = self._map_a(a0, step)
            else:
                item, ws = self._map_b(b0, step - n_a)
                if step == n_a:
                    self._grid_barrier(cnt, epoch, tid, grid)
                    cur_item = Int32(-1)
            if item != cur_item:
                cur_item = item
                cur_e = self._item_expert(base, item, d)
                pass_idx = Int32(0)
                if item >= d:
                    pass_idx = Int32(1)
                self._gather_pass(ids, tw, base, tid, cur_e, pass_idx, routes)
                cute.arch.sync_threads()
                n_rows = Int32(ld_shared_u32(base + Int32(g["off_nrows"])))
                if step < n_a:
                    self._quantize_rows(x, in_gs, base, tid, cur_e, n_rows)
                else:
                    self._load_h(hbuf, base, tid, n_rows)
                    cute.arch.sync_threads()
                    self._quantize_intermediate(fc2_gs, base, tid, cur_e)

            nxt = step + Int32(stages - 1)
            if nxt < total:
                self._issue_step(w13, sf13, w2, sf2, base, tid, d, a0, b0, n_a, nxt, nxt % Int32(stages))
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(stages - 1)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()

            slot_base = base + Int32(g["off_ring"]) + (step % Int32(stages)) * Int32(g["slot_bytes"])
            sf_base = slot_base + Int32(g["slot_payload"])
            if step < n_a:
                self._fc1_step(alpha, hbuf, base, tid, warp, q, c, ws, cur_e, slot_base, sf_base)
            else:
                self._fc2_step(dalpha, out, base, warp, q, c, ws, cur_e, n_rows, slot_base, sf_base)
            cute.arch.sync_threads()
            step += Int32(1)

        # A CTA with no phase-B steps still has to arrive at the barrier.
        if n_b == Int32(0):
            self._grid_barrier(cnt, epoch, tid, grid)
        cute.arch.cp_async_wait_group(0)
