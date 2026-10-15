"""Large-row A16 path for narrow-N, long-K MXFP8 weights: dequantize to BF16, then one BF16 GEMM.

The A16 kernel only has 16-row M tiles (``dense_gemm`` weight-only check: ``mma_tiler_mn`` (16, 64) or (16, 128)), so
at prefill rows it re-dequantizes the whole weight once per 16 rows. For the Qwen3.8-Flash-Next hyperconnection down
projection (K10240 -> N336, a 3.4 MB weight served A16 by its plan record) that leaves the GEMM at ~66 GB/s of
activation reads at M 6048 (k96: 1.883 ms). Above ``MIN_ROWS`` live rows this path dequantizes the weight to BF16
(exact: E4M3 x 2^e8m0 fits BF16) and runs one cuBLAS BF16 GEMM with FP32 accumulation and FP32 output, rounded once
to BF16: the same math as the A16 kernel up to the order of the FP32 additions, so outputs match within FP32
rounding, not byte for byte.

Only the capacity regime of a qualifying query takes it (``engaged``): MXFP8, unpadded K >= 8192, N <= 512, no
declared exact M, low-SM device. Exact-M (decode) regimes, graph capture, preparation calls (priming and selection
races, so autotune still compares the A16 configs on the kernel), rows below ``MIN_ROWS`` and every other shape run the
A16 kernel as before. Plan declarations, tuning knobs, queries and workspace sizes are unchanged, so
plan records and plan population stay the same. torch only: importable without CUDA for the CPU tests.
"""
from __future__ import annotations

import torch

# ponytail: one constant from the k100 probe (M=1024 ties the A16 kernel, M=1264 is 1.2x faster); a per-shape table if a second
# shape ever qualifies with a different crossover.
MIN_ROWS = 1264
MIN_K = 8192
MAX_N = 512
LOW_SM_MAX = 64  # b12x._lib.dense_gemm._DENSE_LOW_SM_MAX_SMS (not imported: it pulls in cutlass)


def engaged(query, sm_count: int) -> bool:
    """Whether a prepared state of ``query`` routes rows >= MIN_ROWS through the BF16 path."""
    return (
        query.recipe == "mxfp8" and query.expected_m is None
        and query.in_features == query.padded_in_features and query.padded_in_features >= MIN_K
        and query.out_features <= MAX_N and int(sm_count) <= LOW_SM_MAX
    )


def unswizzle_scales(scale_bytes: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """F8_128x4 physical UE8M0 storage (flat bytes) -> row-major [n, k // 32] uint8."""
    rows, cols = -(-n // 128) * 128, -(-(k // 32) // 4) * 4
    # inverse of intrinsics.swizzle_block_scale: storage (row tile, col tile, 32, 4, 4) <- logical (rt, 4, 32, ct, 4)
    s = scale_bytes.view(torch.uint8).reshape(rows // 128, cols // 4, 32, 4, 4).permute(0, 3, 2, 1, 4)
    return s.reshape(rows, cols)[:n, : k // 32]


def dequantize(values: torch.Tensor, scale_bytes: torch.Tensor) -> torch.Tensor:
    """MXFP8 [n, k] E4M3 values + F8_128x4 UE8M0 scales -> BF16 [n, k], exact for quantizer output (a NaN scale byte
    0xFF gives inf, and E4M3 subnormals under scales below ~2^-124 underflow in BF16; neither comes out of a quantizer)."""
    n, k = values.shape
    scale = torch.exp2(unswizzle_scales(scale_bytes, n, k).float() - 127).to(torch.bfloat16)
    return (values.view(torch.float8_e4m3fn).to(torch.bfloat16).view(n, k // 32, 32) * scale[:, :, None]).view(n, k)


def run(source: torch.Tensor, values: torch.Tensor, scale_bytes: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """out[m, n] = source[m, k] @ dequantize(values, scales).T, FP32 accumulation and output, one BF16 rounding."""
    weight = dequantize(values, scale_bytes)
    out.copy_(torch.mm(source.reshape(-1, source.shape[-1]), weight.t(), out_dtype=torch.float32).view(out.shape))
    return out
