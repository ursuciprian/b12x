from __future__ import annotations

from dataclasses import replace
import gc
import weakref

import pytest
import torch

from b12x.moe import fused_moe
from b12x._lib.intrinsics import swizzle_block_scale



def weight_plan(**kwargs):
    return fused_moe.plan_weights(
        source=kwargs.pop("source", fused_moe.PackedSource(
            format=fused_moe.PackedSourceFormat.MODELOPT_NVFP4,
            w13_layout=fused_moe.W13Layout.W13,
        )),
        geometry=fused_moe.MoEGeometry(num_experts=8, hidden_size=128, intermediate_size=128),
        activation=kwargs.pop("activation", fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.AUTO, nonlinearity="silu", io_dtype=torch.bfloat16,
        )), **kwargs,
    )


@pytest.mark.parametrize("field,value", [
    ("io_dtype", torch.float16), ("nonlinearity", "relu2"),
])
def test_auto_rejects_unsupported_activation(field, value):
    activation = replace(weight_plan().activation, **{field: value})
    with pytest.raises(ValueError, match="automatic MoE precision"):
        weight_plan(activation=activation)


def test_auto_requires_nvfp4_native_storage():
    plan = weight_plan()
    assert plan._impl.quant_modes == {"nvfp4", "w4a16"}
    assert plan.prepared_format.packing is fused_moe.WeightPacking.SOURCE_NATIVE
    with pytest.raises(ValueError, match="up/gate"):
        weight_plan(source=replace(plan.source, w13_layout=fused_moe.W13Layout.W31))
    with pytest.raises(ValueError, match="source-native"):
        weight_plan(constraints=fused_moe.WeightPlanConstraints(
            required_packing=fused_moe.WeightPacking.MMA_PACKED,
        ))
    with pytest.raises(ValueError, match="ModelOpt NVFP4"):
        weight_plan(source=fused_moe.PackedSource(
            format=fused_moe.PackedSourceFormat.MXFP4_E8M0_K32,
        ))


def test_uniform_nvfp4_a16_requires_only_mma_packing():
    activation = replace(weight_plan().activation, mode=fused_moe.ActivationMode.A16)
    plan = weight_plan(activation=activation)
    assert plan.prepared_format.available_packings == {fused_moe.WeightPacking.MMA_PACKED}
    with pytest.raises(ValueError, match="uniform NVFP4 W4A16 requires mma_packed"):
        weight_plan(activation=activation, constraints=fused_moe.WeightPlanConstraints(
            required_packing=fused_moe.WeightPacking.SOURCE_NATIVE,
        ))


@pytest.mark.parametrize("mode,larger_recipe", (("a4", "nvfp4"), ("a8", "w4a8_nvfp4"), ("auto", "nvfp4_auto")))
def test_a16_cutoff_uses_existing_precision_candidates(mode, larger_recipe):
    from types import SimpleNamespace
    from b12x.moe.fused_moe._impl import B12XFP4ExpertWeights, _PreparedWeightRepresentation
    from b12x.preparation import DeviceIdentity

    base = weight_plan()
    plan = weight_plan(
        activation=replace(base.activation, mode=mode, a16_max_tokens=32),
        source=replace(base.source, w13_layout="w13" if mode == "auto" else "w31"),
    )
    assert plan._impl.w13_layout == "w13"
    assert plan._impl.quant_modes == {"w4a16", "nvfp4" if mode == "auto" else larger_recipe}
    w1 = torch.empty((8, 256, 64), dtype=torch.uint8)
    w2 = torch.empty((8, 128, 64), dtype=torch.uint8)
    experts = fused_moe.PreparedExperts(plan=plan, _impl=B12XFP4ExpertWeights(
        plan=plan._impl,
        w1_fp4=w1, w2_fp4=w2,
        w1_blockscale=torch.empty((8, 256, 8), dtype=torch.uint8),
        w2_blockscale=torch.empty((8, 128, 8), dtype=torch.uint8),
        w1_alphas=torch.ones(8), w2_alphas=torch.ones(8),
        a1_gscale=torch.ones(8), a2_gscale=torch.ones(8),
        representation=_PreparedWeightRepresentation(
            quant_mode="w4a16", layout="source_native",
            value=SimpleNamespace(w13=w1, w2=w2),
        ),
    ))
    declaration = fused_moe.plan_execution(
        experts=experts, capacity=fused_moe.ExecutionCapacity(
            max_tokens=128, top_k=2, warmup_token_counts=(4, 33),
        ),
    )
    assert declaration.token_counts == (4, 32, 33, 128)
    device = DeviceIdentity("nvidia", (12, 0), 188, "SM120")
    for rows, child in declaration.variants.items():
        assert child.query.quant_mode == ("w4a16" if rows <= 32 else larger_recipe)
        configs = [config for _, config in child.contract.eligible_plan(child.query, device).candidates]
        assert configs
        if rows <= 32:
            assert all(config.backend == "w4a16" for config in configs)
        elif mode != "auto":
            assert all(config.backend != "w4a16" for config in configs)


@pytest.mark.parametrize("name", ["w13_blockscale", "w2_blockscale"])
@pytest.mark.parametrize("invalid", ["truncated", "strided", "dtype"])
def test_native_nvfp4_preparation_validates_scale_storage(name, invalid):
    from b12x.moe._shared.kernels.w4a16.prepare import prepare_w4a16_modelopt_native_weights
    inputs = dict(
        w13_fp4=torch.empty((1, 256, 64), dtype=torch.uint8),
        w2_fp4=torch.empty((1, 128, 64), dtype=torch.uint8),
        w13_global_scale=torch.ones(1), w2_global_scale=torch.ones(1),
        w13_blockscale=torch.empty((1, 256, 8), dtype=torch.uint8),
        w2_blockscale=torch.empty((1, 128, 8), dtype=torch.uint8),
        activation="silu",
    )
    scales = inputs[name]
    if invalid == "truncated":
        inputs[name] = scales[:, :-1].contiguous()
    elif invalid == "strided":
        inputs[name] = scales.transpose(1, 2)
    else:
        inputs[name] = scales.float()
    with pytest.raises((ValueError, TypeError), match=name):
        prepare_w4a16_modelopt_native_weights(**inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_uniform_nvfp4_a16_does_not_retain_source_scales():
    plan = weight_plan(activation=replace(weight_plan().activation, mode=fused_moe.ActivationMode.A16))
    weights = fused_moe.PackedWeights(
        w13=torch.zeros((8, 256, 64), dtype=torch.uint8, device="cuda"),
        w2=torch.zeros((8, 128, 64), dtype=torch.uint8, device="cuda"),
        w13_block_scales=swizzle_block_scale(torch.ones((8, 256, 8), device="cuda").to(torch.float8_e4m3fn)),
        w2_block_scales=swizzle_block_scale(torch.ones((8, 128, 8), device="cuda").to(torch.float8_e4m3fn)),
        w13_global_scales=torch.ones(8, device="cuda"),
        w2_global_scales=torch.ones(8, device="cuda"),
    )
    source_scales = (weakref.ref(weights.w13_block_scales), weakref.ref(weights.w2_block_scales))
    experts = fused_moe.prepare_weights(plan=plan, weights=weights)
    prepared = experts._impl.representation_for("w4a16")
    assert prepared.weight_layout == "packed"
    assert prepared.w13.data_ptr() == weights.w13.data_ptr()
    assert prepared.w2.data_ptr() == weights.w2.data_ptr()
    assert prepared.w13_scale is experts._impl.w1_blockscale
    assert prepared.w2_scale is experts._impl.w2_blockscale
    del weights
    gc.collect()
    assert all(ref() is None for ref in source_scales)


