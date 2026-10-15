"""Large-row A16 BF16 path (_a16_dequant): CPU tests for the scale layout, the exact dequantization and the routing
predicate; one GPU test (skipped without CUDA) that runs it through plan_regimes against the A16 kernel."""
from types import SimpleNamespace

import pytest
import torch

from b12x.gemm.blockscaled import _a16_dequant as D

GB10_SMS, RTX_SMS = 48, 188


def _swizzle(scale):
    """b12x._lib.intrinsics.swizzle_block_scale for one 2D scale, flattened (copied: intrinsics imports cutlass)."""
    rows, cols = scale.shape
    rp, cp = -(-rows // 128) * 128, -(-cols // 4) * 4
    padded = torch.zeros((rp, cp), dtype=scale.dtype)
    padded[:rows, :cols] = scale
    return padded.reshape(rp // 128, 4, 32, cp // 4, 4).permute(0, 3, 2, 1, 4).contiguous().view(-1)


def _query(k=10240, n=336, padded=None, expected_m=None, rows=8192):
    # the fields engaged() reads; a real BlockscaledQuery snapshots codegen, which imports cutlass
    return SimpleNamespace(recipe="mxfp8", num_tokens=rows, in_features=k, padded_in_features=padded or k,
                           out_features=n, expected_m=expected_m)


@pytest.mark.parametrize("n, k", ((336, 10240), (130, 320), (8, 128)))
def test_unswizzle_inverts_the_f8_128x4_layout(n, k):
    scale = torch.randint(0, 255, (n, k // 32), dtype=torch.uint8)
    assert torch.equal(D.unswizzle_scales(_swizzle(scale), n, k), scale)


def test_dequantize_is_exact():
    g = torch.Generator().manual_seed(100)
    n, k = 40, 256
    values = (torch.randn(n, k, generator=g) * 64).to(torch.float8_e4m3fn)
    scale = torch.randint(110, 140, (n, k // 32), generator=g, dtype=torch.uint8)
    want = values.double() * torch.exp2(scale.double() - 127).repeat_interleave(32, dim=1)
    got = D.dequantize(values, _swizzle(scale))
    assert got.dtype == torch.bfloat16 and torch.equal(got.double(), want)


@pytest.mark.parametrize("query, sms, want", (
    (_query(), GB10_SMS, True),                          # HC down K10240 -> N336, capacity regime
    (_query(k=8192, n=512), GB10_SMS, True),             # bounds included
    (_query(k=10240, n=520), GB10_SMS, False),           # wider N
    (_query(k=6144, n=2560), GB10_SMS, False),           # GDN out_proj / attention o_proj (tile-fix shape)
    (_query(k=320, n=10240, padded=384), GB10_SMS, False),  # HC up (padded K, served quantized)
    (_query(expected_m=40, rows=40), GB10_SMS, False),   # exact-M decode regime
    (_query(), RTX_SMS, False),                          # high-SM part
))
def test_engaged_only_for_narrow_long_k_capacity_on_low_sm(query, sms, want):
    assert D.engaged(query, sms) is want


def test_preparation_calls_are_flagged_so_races_keep_the_kernel():
    from b12x.preparation.types import call_scope, in_preparation_call
    assert not in_preparation_call()
    with call_scope():
        assert in_preparation_call() and torch.is_inference_mode_enabled()
    assert not in_preparation_call()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_plan_regimes_routes_large_rows_through_bf16_and_keeps_the_kernel_below():
    from b12x.gemm import blockscaled
    from b12x.gemm.blockscaled._tuning import BlockscaledConfig, BlockscaledQuery
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared

    torch.manual_seed(100)
    n, k, cap = 336, 10240, 8192
    w = (torch.randn(n, k, device="cuda") / 4).to(torch.float8_e4m3fn)
    s = (torch.randint(-3, 3, (n, k // 32), device="cuda") + 127).to(torch.uint8)
    packed = blockscaled.pack_weight(w, s)
    query = BlockscaledQuery(recipe="mxfp8", num_tokens=cap, in_features=k, padded_in_features=k, out_features=n,
                             workspace_form="provided", workspace_nbytes=2_000_000_000)
    plan = blockscaled.plan_regimes(query, exact_m=(1, 5, 40),
                                    override=BlockscaledConfig(mode="a16", tile_n=128, tile_k=64, split_k=1))
    dev = torch.device("cuda", torch.cuda.current_device())
    x = torch.randn(cap, k, device="cuda", dtype=torch.bfloat16) / 4
    def factory(rows):  # as vLLM's B12xBlockscaledLinear._call_factory (k90 gemm_probe.py)
        def prepare(st):
            scratch = torch.empty(st.required_workspace, dtype=torch.uint8, device="cuda") if st.required_workspace else None
            return PreparedCall(run=lambda: st.run(x[:rows], packed.weight.values, packed.weight.scale_mma, None,
                                                   activation_scale=None, workspace=scratch))
        return prepare
    with PreparationSession(device=dev, autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(name="k100.test", prepare_calls={r: factory(r) for r in plan.token_counts}),))
        state = require_prepared(plan, "gemm.blockscaled_precision", dev)
        assert state.capacity.a16_dequant and not any(st.a16_dequant for st in state.exact.values())
        ws = torch.empty(max(state.required_workspace, 1), dtype=torch.uint8, device="cuda")
        mm = lambda m: blockscaled.mm(x[:m], packed, plan=plan, workspace=ws)
        got = {m: mm(m) for m in (5, 40, 512, D.MIN_ROWS, 6048)}
        saved, D.MIN_ROWS = D.MIN_ROWS, 1 << 30   # every row count through the A16 kernel
        try:
            kernel = {m: mm(m) for m in got}
        finally:
            D.MIN_ROWS = saved
    weight = w.float() * torch.exp2(s.float() - 127).repeat_interleave(32, dim=1)
    for m, out in got.items():
        if m < D.MIN_ROWS:
            assert torch.equal(out, kernel[m]), m
            continue
        a, b = out.float(), kernel[m].float()
        assert torch.isfinite(a).all()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        rel = ((a - b).abs().max() / b.abs().max()).item()
        assert cos >= 0.99999 and rel <= 1e-2, (m, cos, rel)
        ref = x[:m].float() @ weight.T
        err = lambda t: ((t - ref).norm() / ref.norm()).item()
        assert err(a) <= 1.10 * err(b) + 1e-4, (m, err(a), err(b))
