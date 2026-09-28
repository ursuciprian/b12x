#!/usr/bin/env python3
"""Read-bandwidth ceiling of cp.async streaming on this GPU, with no compute.

Each CTA streams its share of a 1 GB buffer into a shared-memory ring of STAGES
slots of CHUNK bytes with 16-byte cp.async.cg, the same copy style as the MoE decode
kernels, and does nothing with the data. The sweep covers threads per CTA, ring
depth, CTAs per SM, and the contiguous span each CTA owns per stage. torch's
int32 sum over the same buffer is the library reference. Every kernel reads the
full buffer once per call, so L2 cannot hold the working set.

    python benchmarks/benchmark_stream_ceiling.py
"""

from __future__ import annotations


import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32, Int64

from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    cp_async_u64_shared_global,
    cp_async_bulk_g2s_mbar,
    get_ptr_as_int64,
    shared_ptr_to_u32,
)
from b12x._lib.utils import current_cuda_stream, make_ptr

BYTES = 1 << 30


class StreamProbe:
    def __init__(self, threads, stages, chunk, interleave=False, gather4=0, sf=""):
        self.threads, self.stages, self.chunk = threads, stages, chunk
        self.interleave, self.gather4, self.sf = interleave, gather4, sf
        extra = 4 * gather4 + (1280 if sf else 0)
        self.words = stages * (chunk + extra) // 4
        self.slot = chunk + extra

    @cute.jit
    def __call__(self, src_ptr: cute.Pointer, sf_ptr: cute.Pointer, n_chunks: Int32, grid: Int32, stream):
        src = cute.make_tensor(src_ptr, cute.make_layout(Int64(BYTES)))
        sfr = cute.make_tensor(sf_ptr, cute.make_layout(Int64(BYTES // 8)))
        self.kernel(src, sfr, n_chunks).launch(grid=(grid, 1, 1), block=[self.threads, 1, 1], stream=stream)

    @cute.jit
    def _issue_sf(self, sfr, dst, tid, chunk_idx):
        # 1280 scale bytes per stage from a separate plane (the real F8_128x4 addressing):
        #   g4  = today's wm FC1 gather: 8 consecutive rows x 40 K64, 4 B each (320 copies)
        #   g8  = row pairs (x, x+32) share 8 B: 4 x's x 40 K64 (160 copies)
        #   c16 = a linear [row][K64] plane: 80 contiguous 16 B copies
        atom = Int64(chunk_idx // Int32(16)) * Int64(20480)
        sub = chunk_idx % Int32(16)
        if cutlass.const_expr(self.sf == "g4"):
            for i in cutlass.range_constexpr(-(-320 // self.threads)):
                w = tid + Int32(i * self.threads)
                if w < Int32(320):
                    r, k64 = w // Int32(40), w % Int32(40)
                    off = atom + Int64(k64 * Int32(512) + ((sub % Int32(4)) * Int32(8) + r) * Int32(16) + (sub // Int32(4)) * Int32(4))
                    cp_async_u32_shared_global(dst + w * Int32(4), get_ptr_as_int64(sfr, off))
        if cutlass.const_expr(self.sf == "g8"):
            for i in cutlass.range_constexpr(-(-160 // self.threads)):
                w = tid + Int32(i * self.threads)
                if w < Int32(160):
                    k64, x = w // Int32(4), w % Int32(4)
                    off = atom + Int64(k64 * Int32(512) + ((sub % Int32(8)) * Int32(4) + x) * Int32(16) + (sub // Int32(8)) * Int32(8))
                    cp_async_u64_shared_global(dst + w * Int32(8), get_ptr_as_int64(sfr, off))
        if cutlass.const_expr(self.sf == "c16"):
            if tid < Int32(80):
                cp_async4_shared_global(dst + tid * Int32(16),
                                        get_ptr_as_int64(sfr, Int64(chunk_idx) * Int64(1280) + Int64(tid * Int32(16))))

    @cute.jit
    def _issue(self, src, sfr, base, tid, first, step, slot, per_thread):
        chunk_idx = first + step
        if cutlass.const_expr(self.interleave):
            # Alternate between the CTA's region halves, like wm's up/gate/down streams.
            # Two streams 400 KB apart, each 40 chunks long, every chunk read once.
            half = step & Int32(1)
            j = step >> Int32(1)
            chunk_idx = first + ((j // Int32(40)) * Int32(2) + half) * Int32(40) + j % Int32(40)
        dst = base + slot * Int32(self.slot)
        for i in cutlass.range_constexpr(per_thread):
            off = Int64(chunk_idx) * Int64(self.chunk) + Int64((tid + Int32(i * self.threads)) * Int32(16))
            cp_async4_shared_global(dst + (tid + Int32(i * self.threads)) * Int32(16), get_ptr_as_int64(src, off))
        for i in cutlass.range_constexpr(-(-self.gather4 // self.threads)):
            w = tid + Int32(i * self.threads)
            if w < Int32(self.gather4):
                # wm-like scale gather: 4 B at a 512 B stride per K64 group, 16 B per row.
                off = Int64(chunk_idx) * Int64(self.chunk) + Int64((w % Int32(40)) * Int32(512) + (w // Int32(40)) * Int32(16))
                cp_async_u32_shared_global(dst + Int32(self.chunk) + w * Int32(4), get_ptr_as_int64(src, off))
        if cutlass.const_expr(self.sf != ""):
            self._issue_sf(sfr, dst + Int32(self.chunk), tid, chunk_idx)

    @cute.kernel
    def kernel(self, src: cute.Tensor, sfr: cute.Tensor, n_chunks: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        tid, bid, grid = Int32(tidx), Int32(bidx), Int32(gdim)
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[cute.struct.MemRange[cutlass.Uint32, self.words], 1024]

        base = shared_ptr_to_u32(smem.allocate(Storage).words.data_ptr())
        per_cta = n_chunks // grid
        first = bid * per_cta
        per_thread = self.chunk // 16 // self.threads

        for s0 in cutlass.range_constexpr(self.stages - 1):
            if Int32(s0) < per_cta:
                self._issue(src, sfr, base, tid, first, Int32(s0), Int32(s0), per_thread)
            cute.arch.cp_async_commit_group()
        step = Int32(0)
        while step < per_cta:
            nxt = step + Int32(self.stages - 1)
            if nxt < per_cta:
                slot = nxt % Int32(self.stages)
                self._issue(src, sfr, base, tid, first, nxt, slot, per_thread)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(self.stages - 1)
            cute.arch.sync_threads()
            step += Int32(1)
        cute.arch.cp_async_wait_group(0)


class TmaStreamProbe:
    """One bulk copy (cp.async.bulk + mbarrier) of CHUNK bytes per stage, issued by thread 0."""

    def __init__(self, threads, stages, chunk):
        self.threads, self.stages, self.chunk = threads, stages, chunk
        self.words = stages * chunk // 4

    @cute.jit
    def __call__(self, src_ptr: cute.Pointer, n_chunks: Int32, grid: Int32, stream):
        src = cute.make_tensor(src_ptr, cute.make_layout(Int64(BYTES)))
        self.kernel(src, n_chunks).launch(grid=(grid, 1, 1), block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, src: cute.Tensor, n_chunks: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        tid, bid, grid = Int32(tidx), Int32(bidx), Int32(gdim)
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            mbar: cute.struct.MemRange[cutlass.Int64, self.stages]
            words: cute.struct.Align[cute.struct.MemRange[cutlass.Uint32, self.words], 1024]

        storage = smem.allocate(Storage)
        base = shared_ptr_to_u32(storage.words.data_ptr())
        mbar = storage.mbar.data_ptr()
        if tid == Int32(0):
            for st in cutlass.range_constexpr(self.stages):
                cute.arch.mbarrier_init(mbar + st, Int32(1))
        cute.arch.sync_threads()
        per_cta = n_chunks // grid
        first = bid * per_cta
        if tid == Int32(0):
            for s0 in cutlass.range_constexpr(self.stages - 1):
                if Int32(s0) < per_cta:
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar + s0, Int32(self.chunk))
                    cp_async_bulk_g2s_mbar(base + Int32(s0 * self.chunk),
                                           get_ptr_as_int64(src, Int64(first + Int32(s0)) * Int64(self.chunk)),
                                           Int32(self.chunk), shared_ptr_to_u32(mbar + s0))
        step = Int32(0)
        while step < per_cta:
            nxt = step + Int32(self.stages - 1)
            if tid == Int32(0):
                if nxt < per_cta:
                    slot = nxt % Int32(self.stages)
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar + slot, Int32(self.chunk))
                    cp_async_bulk_g2s_mbar(base + slot * Int32(self.chunk),
                                           get_ptr_as_int64(src, Int64(first + nxt) * Int64(self.chunk)),
                                           Int32(self.chunk), shared_ptr_to_u32(mbar + slot))
            cur = step % Int32(self.stages)
            cute.arch.mbarrier_wait(mbar + cur, phase=(step // Int32(self.stages)) & Int32(1))
            cute.arch.sync_threads()
            step += Int32(1)


def timed(fn, reps=10):
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(reps):
        a.record()
        fn()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) / 1e3)
    out.sort()
    return out[len(out) // 2]


def main():
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    buf = torch.randint(0, 1 << 30, (BYTES // 4,), dtype=torch.int32, device=dev)
    t = timed(lambda: buf.sum())
    print(f"torch int32 sum: {BYTES / t / 1e9:.1f} GB/s  (SMs={sms})", flush=True)
    t = timed(lambda: buf.view(torch.float32).amax())
    print(f"torch f32 amax : {BYTES / t / 1e9:.1f} GB/s", flush=True)
    ptr = make_ptr(cutlass.Uint8, buf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    sfbuf = torch.zeros(BYTES // 8, dtype=torch.uint8, device=dev)
    sptr = make_ptr(cutlass.Uint8, sfbuf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    print(f"{'mode':>18} {'grid':>5} {'GB/s':>7}")
    chunk, stages = 10240, 5
    import os
    modes = (("seq", False, 0, ""), ("interleave", True, 0, ""),
             ("seq+gather320", False, 320, ""), ("interleave+gather320", True, 320, ""),
             ("seq+sf:g4", False, 0, "g4"), ("seq+sf:g8", False, 0, "g8"), ("seq+sf:c16", False, 0, "c16"))
    only = os.environ.get("PROBE_MODES")
    for mode, interleave, gather, sf in modes:
        if only and mode not in only.split(","):
            continue
        probe = StreamProbe(128, stages, chunk, interleave=interleave, gather4=gather, sf=sf)
        for grid in (16, 24, 33, 48):
            n_chunks = BYTES // chunk
            n_chunks -= n_chunks % (grid * 80)
            compiled = cute.compile(probe, ptr, sptr, Int32(n_chunks), Int32(grid), current_cuda_stream())
            t = timed(lambda: compiled(ptr, sptr, Int32(n_chunks), Int32(grid), current_cuda_stream()))
            print(f"{mode:>18} {grid:>5} {n_chunks * chunk / t / 1e9:>7.1f}", flush=True)

if __name__ == "__main__":
    main()
