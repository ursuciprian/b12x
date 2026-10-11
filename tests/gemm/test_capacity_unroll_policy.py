"""MXFP8 capacity-regime plan for large prefill chunks (CPU only: host tile/policy selection, no kernel).

A plan without expected_m declared for >= 4096 rows with K >= 4096 (vLLM's max_num_batched_tokens capacity regime) on
low-SM parts gets 128x128/BK64 for wide N and the persistent swizzle (large-M unroll off); every other regime keeps
its previous choice.
"""
import cutlass

from b12x._lib.dense_gemm import (
    _dense_gemm_policy_for, _select_default_dense_gemm_plan, _select_mxfp8_tile_k,
)


def _unroll(m, n, k, *, expected_m=None, sm_count=48, tile=(64, 128), tile_k=128, mxfp8=True):
    return _dense_gemm_policy_for(
        m=m, n=n, k=k, l=1, ab_dtype=cutlass.Float8E4M3FN, c_dtype=cutlass.BFloat16,
        mma_tiler_mn=tile, cluster_shape_mn=(1, 1), sm_count=sm_count, tile_k=tile_k,
        expected_m=expected_m, generalize_mxfp8_split_k=mxfp8,
    ).large_m_unroll


def _plan(m, n, k, *, expected_m=None, sm_count=48):
    tile = tuple(_select_default_dense_gemm_plan(m, n, k, sm_count, is_mxfp8=True, expected_m=expected_m).mma_tiler_mn)
    return tile, _select_mxfp8_tile_k(m, n, k, expected_m, sm_count)


def test_capacity_regime_large_rows_long_k_uses_bk64_without_unroll():
    assert _plan(8192, 2560, 6144) == ((128, 128), 64)
    assert _plan(9072, 2560, 6144) == ((128, 128), 64)
    assert _plan(4096, 4096, 4096) == ((128, 128), 64)
    assert _unroll(8192, 2560, 6144, tile=(128, 128), tile_k=64) is False
    assert _unroll(8192, 2560, 6144) is False


def test_other_regimes_keep_their_choice():
    assert _plan(8192, 16384, 2560) == ((64, 128), 128)       # short K
    assert _plan(2048, 2560, 6144) == ((128, 64), 128)        # small capacity (21e0b201 choice)
    assert _plan(8192, 1024, 6144)[0] == (64, 64)             # narrow N
    assert _plan(8192, 2560, 6144, sm_count=160)[1] == 128    # high-SM parts
    assert _plan(6, 2560, 6144, expected_m=6)[0] == (16, 128)  # exact decode tile
    assert _unroll(8192, 16384, 2560) is True
    assert _unroll(2048, 2560, 6144) is True
    assert _unroll(6048, 2560, 6144, expected_m=6048) is True
    assert _unroll(6, 2560, 6144, expected_m=6) is False
    assert _unroll(8192, 2560, 6144, sm_count=160) is True
    assert _unroll(8192, 2560, 6144, mxfp8=False) is True
