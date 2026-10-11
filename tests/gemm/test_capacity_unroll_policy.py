"""Large-M unroll policy of the MXFP8 capacity regime (CPU only: host policy, no kernel).

A plan without expected_m declared for >= 4096 rows (vLLM's max_num_batched_tokens capacity regime) keeps the
16-way persistent swizzle when K >= 4096; every other regime keeps its previous choice.
"""
import cutlass

from b12x._lib.dense_gemm import _dense_gemm_policy_for


def _unroll(m, n, k, *, expected_m=None, sm_count=48, tile=(64, 128), tile_k=128, mxfp8=True):
    return _dense_gemm_policy_for(
        m=m, n=n, k=k, l=1, ab_dtype=cutlass.Float8E4M3FN, c_dtype=cutlass.BFloat16,
        mma_tiler_mn=tile, cluster_shape_mn=(1, 1), sm_count=sm_count, tile_k=tile_k,
        expected_m=expected_m, generalize_mxfp8_split_k=mxfp8,
    ).large_m_unroll


def test_capacity_regime_large_rows_long_k_turns_unroll_off():
    assert _unroll(8192, 2560, 6144) is False
    assert _unroll(9072, 2560, 6144) is False
    assert _unroll(4096, 2560, 4096) is False


def test_other_regimes_keep_their_choice():
    assert _unroll(8192, 16384, 2560) is True          # short K
    assert _unroll(2048, 2560, 6144) is True           # small capacity
    assert _unroll(6048, 2560, 6144, expected_m=6048) is True   # exact large M (declared regime)
    assert _unroll(6, 2560, 6144, expected_m=6) is False       # exact decode
    assert _unroll(8192, 2560, 6144, sm_count=160) is True      # high-SM parts
    assert _unroll(8192, 2560, 6144, mxfp8=False) is True       # block/plain FP8 policy unchanged
