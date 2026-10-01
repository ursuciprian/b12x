"""Weight-only MXFP8 GEMV for decode-sized row counts (SM120/SM121).

The A16 tensor-core path pads up to eight live rows to a 16-row MMA tile and
launches one CTA per (N tile, K split). At decode sizes the step is bound by
weight bytes, and small grids leave SMs idle (N=2560 at tile 64 is 40 CTAs on
48 SMs). This kernel streams the shared packed weight exactly once through a
persistent, SM-sized grid:

* every lane group of ``lanes`` threads owns one weight row (output column) per
  pass and reads its K slice in 16-byte E4M3 chunks, all loads of a pass issued
  before any arithmetic so each lane keeps several requests in flight;
* the CTA stages the BF16 activations of its K slice in shared memory once,
  rows past the live count zero-filled, so compute carries no row predicate;
* each 16-value partial dot is accumulated in FP32 and then multiplied by its
  UE8M0 group scale, which is an exact power of two;
* ``split`` > 1 writes FP32 partials that the existing A16 split-K reduction
  sums, which keeps the workspace contract of the A16 path.

Live rows are a runtime argument; ``rows`` is the plan's capacity (<= 8).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile, run_compiled
from b12x._lib.intrinsics import (
    fp8x4_e4m3_to_bfloat2x2_native_sm120,
    get_ptr_as_int64,
    ld_global_nc_u32,
    ld_global_nc_v4_u32,
    ld_global_v4_u32,
    ld_shared_v4_u32,
    shared_ptr_to_u32,
    st_shared_v4_u32,
    u32_as_f32,
    warp_reduce,
)
from b12x._lib.program_cache import program_cache
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

MAX_ROWS = 8
WARPS = 8
SPLITS = (1, 2, 4, 8)
CTAS_PER_SM = (1, 2, 4)
# Usable shared memory per SM on SM12x (99 KiB opt-in), minus the 1 KiB the
# driver reserves for each resident CTA.
_SMEM_PER_SM = 99 * 1024


def lanes_for(chunks: int) -> int:
    """Largest power-of-two lane group (<= 32) that divides the chunk count."""
    lanes = 32
    while lanes > 1 and chunks % lanes:
        lanes //= 2
    return lanes


def geometry(n: int, k: int, rows: int, split: int, ctas_per_sm: int, sm_count: int):
    """Static launch geometry, or None when the configuration cannot run."""
    if not 0 < rows <= MAX_ROWS or k % (32 * split):
        return None
    ks = k // split
    chunks = ks // 16
    lanes = lanes_for(chunks)
    if chunks // lanes > 24:  # bounded registers for the in-flight loads
        return None
    smem = rows * ks * 2
    if (smem + 1024) * ctas_per_sm > _SMEM_PER_SM:
        return None
    columns_per_warp = 32 // lanes
    tasks = -(-n // columns_per_warp)
    ctas = max(1, min(sm_count * ctas_per_sm, -(-tasks // WARPS)))
    return dict(ks=ks, chunks=chunks, lanes=lanes, ctas=ctas, smem=smem)


def default_split(n: int, k: int, rows: int, sm_count: int) -> int | None:
    """Fewest K splits that fit two CTAs per SM and give each warp two passes;
    the widest feasible split when no split reaches that (small N)."""
    fallback = None
    for split in SPLITS:
        g = geometry(n, k, rows, split, 2, sm_count)
        if g is None:
            continue
        fallback = split
        if -(-n // (32 // g["lanes"])) * split >= 2 * 2 * sm_count * WARPS:
            return split
    return fallback


def _fadd(a, b):
    return a + b


@cute.jit
def _flat(pointer: cute.Pointer):
    return cute.make_tensor(pointer, cute.make_layout((Int64(1) << Int64(50),)))


@cute.jit
def _lo(v: Uint32) -> Float32:
    return u32_as_f32(v << Uint32(16))


@cute.jit
def _hi(v: Uint32) -> Float32:
    return u32_as_f32(v & Uint32(0xFFFF0000))


class Mxfp8SkinnyGemv:
    def __init__(self, n: int, k: int, rows: int, split: int, ctas_per_sm: int, sm_count: int,
                 stored_k: int | None = None):
        g = geometry(n, k, rows, split, ctas_per_sm, sm_count)
        if g is None:
            raise ValueError(f"MXFP8 GEMV cannot run N={n} K={k} rows={rows} split={split} ctas/SM={ctas_per_sm}")
        self.n, self.k, self.rows, self.split = n, k, rows, split
        self.ks, self.chunks, self.lanes, self.ctas = g["ks"], g["chunks"], g["lanes"], g["ctas"]
        self.per_lane = self.chunks // self.lanes
        self.groups = 32 // self.lanes
        # Packed weights may pad K (e.g. to 128); rows and scale tiles use the
        # stored width while the dot covers only the logical K.
        self.stored_k = k if stored_k is None else stored_k
        self.k_tiles = -(-(self.stored_k // 32) // 4)
        self.threads = 32 * WARPS

    @cute.jit
    def __call__(self, source: cute.Pointer, weight: cute.Pointer, scale: cute.Pointer,
                 output: cute.Pointer, m: Int32, stream: cuda.CUstream):
        self.kernel(_flat(source), _flat(weight), _flat(scale), _flat(output), m).launch(
            grid=(self.ctas, self.split, 1), block=(self.threads, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, source: cute.Tensor, weight: cute.Tensor, scale: cute.Tensor,
               output: cute.Tensor, m: Int32):
        thread, _, _ = cute.arch.thread_idx()
        cta, part, _ = cute.arch.block_idx()
        tid = Int32(thread)
        allocator = cutlass.utils.SmemAllocator()
        staged = allocator.allocate_tensor(
            BFloat16, cute.make_layout((self.rows * self.ks,)), byte_alignment=16
        )
        k_base = Int64(part) * Int64(self.ks)

        # Stage this CTA's activation slice; rows past the live count are zero.
        vectors = self.ks // 8
        index = tid
        while index < Int32(self.rows * vectors):
            row = index // Int32(vectors)
            column = index % Int32(vectors)
            v0, v1, v2, v3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
            if row < m:
                v0, v1, v2, v3 = ld_global_v4_u32(get_ptr_as_int64(
                    source, Int64(row) * Int64(self.k) + k_base + Int64(column) * Int64(8)
                ))
            st_shared_v4_u32(shared_ptr_to_u32(staged.iterator + index * Int32(8)), v0, v1, v2, v3)
            index += Int32(self.threads)
        cute.arch.barrier()

        x_base = shared_ptr_to_u32(staged.iterator)
        warp = tid // Int32(32)
        lane = tid % Int32(32)
        group = lane // Int32(self.lanes)
        in_group = lane % Int32(self.lanes)
        first = (Int32(cta) * Int32(WARPS) + warp) * Int32(self.groups)
        stride = Int32(self.ctas * WARPS * self.groups)
        size = Int64(m) * Int64(self.n)
        group_base = Int32(part) * Int32(self.ks // 32)
        acc = cute.make_rmem_tensor((self.rows,), Float32)
        # The loop bound is warp-uniform so every lane reaches the shuffles.
        while first < Int32(self.n):
            column = first + group
            clamped = column
            if column >= Int32(self.n):
                clamped = Int32(self.n - 1)
            w_row = Int64(clamped) * Int64(self.stored_k) + k_base
            s_row = (Int64(clamped // Int32(128)) * Int64(self.k_tiles * 512)
                     + Int64(clamped % Int32(32)) * Int64(16)
                     + Int64((clamped % Int32(128)) // Int32(32)) * Int64(4))
            words = []
            factors = []
            for i in cutlass.range_constexpr(self.per_lane):
                chunk = in_group + Int32(i * self.lanes)
                w0, w1, w2, w3 = ld_global_nc_v4_u32(
                    get_ptr_as_int64(weight, w_row + Int64(chunk) * Int64(16))
                )
                words.append((w0, w1, w2, w3))
                g = group_base + chunk // Int32(2)
                packed = ld_global_nc_u32(get_ptr_as_int64(
                    scale, s_row + Int64(g // Int32(4)) * Int64(512)
                ))
                shift = Uint32(g % Int32(4)) * Uint32(8)
                factors.append(u32_as_f32(((packed >> shift) & Uint32(0xFF)) << Uint32(23)))
            for r in cutlass.range_constexpr(self.rows):
                acc[r] = Float32(0.0)
            for i in cutlass.range_constexpr(self.per_lane):
                chunk = in_group + Int32(i * self.lanes)
                w0, w1, w2, w3 = words[i]
                b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(w0)
                b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(w1)
                b4, b5 = fp8x4_e4m3_to_bfloat2x2_native_sm120(w2)
                b6, b7 = fp8x4_e4m3_to_bfloat2x2_native_sm120(w3)
                wf = (
                    _lo(b0), _hi(b0), _lo(b1), _hi(b1), _lo(b2), _hi(b2), _lo(b3), _hi(b3),
                    _lo(b4), _hi(b4), _lo(b5), _hi(b5), _lo(b6), _hi(b6), _lo(b7), _hi(b7),
                )
                for r in cutlass.range_constexpr(self.rows):
                    address = x_base + (Int32(r * self.ks) + chunk * Int32(16)) * Int32(2)
                    x0, x1, x2, x3 = ld_shared_v4_u32(address)
                    x4, x5, x6, x7 = ld_shared_v4_u32(address + Int32(16))
                    xs = (x0, x1, x2, x3, x4, x5, x6, x7)
                    partial = Float32(0.0)
                    for j in cutlass.range_constexpr(8):
                        partial += wf[2 * j] * _lo(xs[j])
                        partial += wf[2 * j + 1] * _hi(xs[j])
                    acc[r] += partial * factors[i]
            for r in cutlass.range_constexpr(self.rows):
                total = warp_reduce(acc[r], _fadd, self.lanes)
                if column < Int32(self.n):
                    if in_group == Int32(0):
                        if Int32(r) < m:
                            offset = Int64(r) * Int64(self.n) + Int64(column)
                            if cutlass.const_expr(self.split == 1):
                                output[offset] = total.to(BFloat16)
                            else:
                                output[Int64(part) * size + offset] = total
            first += stride


@program_cache
def compile_gemv(n: int, k: int, stored_k: int, rows: int, split: int, ctas_per_sm: int,
                 sm_count: int, device: int):
    """Compile one static GEMV geometry; the live row count stays a launch scalar."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("MXFP8 GEMV must be compiled before CUDA graph capture")
    kernel = Mxfp8SkinnyGemv(n, k, rows, split, ctas_per_sm, sm_count, stored_k)
    key = (n, k, stored_k, rows, split, ctas_per_sm, sm_count, device)
    out_type = BFloat16 if split == 1 else Float32
    with torch.cuda.device(device):
        raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
        raw = b12x_compile(
            kernel,
            make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(out_type, 16, cute.AddressSpace.gmem, assumed_align=16),
            Int32(1), current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key("gemm.blockscaled.mxfp8_gemv", 1, key),
        )

    def run(source: torch.Tensor, values: torch.Tensor, scale_bytes: torch.Tensor,
            target: torch.Tensor, m: int):
        run_compiled(raw, (
            make_ptr(BFloat16, source.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Uint8, values.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Uint8, scale_bytes.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(out_type, target.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            Int32(m), current_cuda_stream(),
        ))

    return attach_programs(run, raw)
