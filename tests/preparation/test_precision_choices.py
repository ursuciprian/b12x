"""Precision and materialization choices retain their numerical prerequisites."""
from dataclasses import replace

import pytest

from b12x.preparation import DeviceIdentity, FrozenMapping


DEVICE = DeviceIdentity("nvidia", (12, 0), 188, "SM120")


@pytest.mark.parametrize("mode", ("auto", "quantized", "a16"))
@pytest.mark.parametrize("cutoff", (0, 32, 128, 256))
def test_dense_a16_cutoff_constrains_candidates_and_covers_unlisted_counts(mode, cutoff):
    from types import SimpleNamespace
    import torch
    from b12x.gemm.blockscaled import BlockscaledQuery, plan_regimes

    query = BlockscaledQuery(
        recipe="nvfp4", num_tokens=128, in_features=256, padded_in_features=256,
        out_features=128, activation_mode=mode, activation_scale_available=True,
    )
    plan = plan_regimes(query, exact_m=(4, 64), a16_max_tokens=cutoff)
    states = {}
    for rows, child in plan.variants.items():
        expected = "a16" if rows <= cutoff else mode
        assert child.query.activation_mode == expected
        candidates = child.contract.eligible_plan(child.query, DEVICE).candidates
        assert candidates
        if expected != "auto":
            assert all(config.mode == expected for _, config in candidates)
        states[rows] = SimpleNamespace(query=child.query, required_workspace=rows)
    state = plan._assemble(states, None)
    for rows in (1, 4, 17, 31, 32, 33, 64, 127, 128):
        selected = state.resolve(torch.empty(rows, 256, device="meta"))
        assert selected.query.activation_mode == ("a16" if rows <= cutoff else mode)
        assert selected.query.num_tokens >= rows
    assert state.required_workspace == 128


@pytest.mark.parametrize("cutoff", (-1, 1.5, True))
def test_a16_cutoff_rejects_invalid_values(cutoff):
    import torch
    from b12x.gemm.blockscaled import BlockscaledQuery, plan_regimes
    from b12x.moe.fused_moe import ActivationMode, ActivationSpec

    query = BlockscaledQuery(
        recipe="nvfp4", num_tokens=128, in_features=256,
        padded_in_features=256, out_features=128,
    )
    with pytest.raises(ValueError, match="nonnegative integer"):
        plan_regimes(query, a16_max_tokens=cutoff)
    with pytest.raises(ValueError, match="nonnegative integer"):
        ActivationSpec(mode=ActivationMode.A4, nonlinearity="silu",
                       io_dtype=torch.bfloat16, a16_max_tokens=cutoff)


@pytest.mark.parametrize("rows", (1, 8, 128))
def test_dense_nvfp4_auto_races_a4_and_a16(rows):
    from b12x.gemm.blockscaled._tuning import BlockscaledQuery, TUNING

    query = BlockscaledQuery(
        recipe="nvfp4", num_tokens=rows, in_features=2560, padded_in_features=2560,
        out_features=512, activation_mode="auto", activation_scale_available=True,
    )
    configs = [config for _, config in TUNING.eligible_plan(query, DEVICE).candidates]
    assert {config.mode for config in configs} == {"a16", "quantized"}
    assert TUNING.configure(query, device=DEVICE).default.mode == ("a16" if rows <= 8 else "quantized")
    for mode in ("a16", "quantized"):
        forced = replace(query, activation_mode=mode)
        assert {config.mode for _, config in TUNING.eligible_plan(forced, DEVICE).candidates} == {mode}
        selected = next(config for config in configs if config.mode == mode)
        assert TUNING.configure(query, device=DEVICE, override=selected).default == selected


def test_dense_nvfp4_without_activation_scale_excludes_a4():
    from b12x.gemm.blockscaled._tuning import BlockscaledQuery, TUNING

    query = BlockscaledQuery(
        recipe="nvfp4", num_tokens=1, in_features=256, padded_in_features=256,
        out_features=128, activation_mode="auto", activation_scale_available=False,
    )
    assert {config.mode for _, config in TUNING.eligible_plan(query, DEVICE).candidates} == {"a16"}


def _nvfp4_query():
    from b12x.moe.fused_moe._tuning import MoeDecodeQuery

    return MoeDecodeQuery(
        quant_mode="nvfp4", quant_modes=("nvfp4",), source_format="modelopt_nvfp4",
        activation="silu", io_dtype="bfloat16", num_experts=32, hidden_size=512,
        intermediate_size=512, top_k=4, num_tokens=1024, routed_rows=4096,
        route_num_experts=32, route_logits_dtype=None, apply_router_weight_on_input=False,
        collect_activation_amax=False, deterministic_output=False, swiglu_limit=None,
        swiglu_alpha=1.0, swiglu_beta=0.0, w13_layout="w13", weight_layouts=("mma_view",),
        w4a16_weight_layout=None, w4a16_scale_format=None, w4a16_block_size_m=None,
        fast_math=True, numerical_recipe=None, controls=FrozenMapping(), shared_input_scales=True,
    )


def test_nvfp4_split_materialization_is_a_real_candidate_and_default():
    from b12x.moe.fused_moe._tuning import TUNING

    query = _nvfp4_query()
    configs = [config for _, config in TUNING.eligible_plan(query, DEVICE).candidates]
    assert {config.nvfp4_materialize_intermediate for config in configs} == {False, True}
    for config in configs:
        if config.nvfp4_materialize_intermediate:
            assert config.nvfp4_share_input and config.dynamic_tile_m == 128
    default = TUNING.configure(query, device=DEVICE).default
    assert default.nvfp4_share_input and default.nvfp4_materialize_intermediate
    assert default in configs


@pytest.mark.parametrize("changes", (
    {"shared_input_scales": False}, {"deterministic_output": True}, {"activation": "relu2"},
    {"intermediate_size": 320}, {"controls": FrozenMapping({"dynamic_down_scale": True})},
    {"controls": FrozenMapping({"dynamic_swap_ab": "1"})},
    {"controls": FrozenMapping({"dynamic_work_source": "ready_queue"})},
))
def test_nvfp4_split_materialization_rejects_unsupported_contracts(changes):
    from b12x.moe.fused_moe._tuning import TUNING

    query = _nvfp4_query()
    split = next(config for _, config in TUNING.eligible_plan(query, DEVICE).candidates
                 if config.nvfp4_materialize_intermediate)
    invalid = replace(query, **changes)
    with pytest.raises(ValueError):
        TUNING.configure(invalid, device=DEVICE, override=split)
    assert all(not config.nvfp4_materialize_intermediate
               for _, config in TUNING.eligible_plan(invalid, DEVICE).candidates)


def test_compact_w4a8_has_only_its_supported_dynamic_route():
    from b12x.moe.fused_moe._tuning import TUNING

    query = replace(_nvfp4_query(), quant_mode="w4a8_mx", quant_modes=("w4a8_mx",),
                    source_format="fp4_e8m0_k32", intermediate_size=576)
    configs = [config for _, config in TUNING.eligible_plan(query, DEVICE).candidates]
    assert configs
    assert {(config.backend, config.dynamic_tile_m, config.dynamic_route_mode, config.route_planner)
            for config in configs} == {("dynamic", 16, "grouped", "internal")}


@pytest.mark.parametrize("heads", (1, 12, 16, 20, 32))
@pytest.mark.parametrize("mode", ("decode", "extend"))
def test_v41_precision_candidates_match_native_head_group_contract(heads, mode):
    import torch
    from b12x.attention import compressed_sparse_mla as mla
    from b12x.attention.compressed_sparse_mla._tuning import TUNING

    q = torch.empty((3, heads, 512), dtype=torch.bfloat16)
    cache = torch.empty((2, 64 * 528), dtype=torch.uint8)
    plan = mla.plan(
        mla.Caps(device="cpu", num_q_heads=heads, max_q_rows=3, max_width=128,
                 swa_width=128, indexed_width=0, cache_format="deepseek_v41", mode=mode),
        invocation=mla.invocation_from_tensors(q=q, swa_k_cache=cache, out=torch.empty_like(q)),
    )
    configs = [config for _, config in TUNING.eligible_plan(plan.query, DEVICE).candidates]
    expected = {("bf16", 16), ("fp8", 16)} if mode == "extend" else {
        ("bf16", 16), ("fp8", 8),
    }
    if mode == "decode" and heads % 16 == 0:
        expected.add(("fp8", 16))
    assert {(config.v41_compute_mode, config.v41_heads_per_block) for config in configs} == expected
    default = TUNING.configure(plan.query, device=DEVICE).default
    assert default in configs and default.v41_compute_mode == "fp8"
    assert default.v41_heads_per_block == (8 if mode == "decode" and heads % 16 else 16)
    for config in configs:
        assert TUNING.configure(plan.query, device=DEVICE, override=config).default == config
    invalid = replace(default, v41_compute_mode="bf16", v41_heads_per_block=8)
    with pytest.raises(ValueError, match="head grouping"):
        TUNING.configure(plan.query, device=DEVICE, override=invalid)
    if mode == "decode" and heads % 16:
        with pytest.raises(ValueError, match="complete 16-head groups"):
            TUNING.configure(plan.query, device=DEVICE, override=replace(default, v41_heads_per_block=16))


@pytest.mark.parametrize("mode", ("decode", "extend"))
def test_compressed_mla_rejects_unaligned_q_before_preparation(mode):
    import torch
    from b12x.attention import compressed_sparse_mla as mla

    storage = torch.empty(3 * 16 * 512 + 1, dtype=torch.bfloat16)
    q = storage[1:].view(3, 16, 512)
    cache = torch.empty((2, 64 * 528), dtype=torch.uint8)
    with pytest.raises(ValueError, match="Q requires 16-byte alignment"):
        mla.plan(
            mla.Caps(device="cpu", num_q_heads=16, max_q_rows=3, max_width=128,
                     swa_width=128, indexed_width=0, cache_format="deepseek_v41", mode=mode),
            invocation=mla.invocation_from_tensors(q=q, swa_k_cache=cache, out=torch.empty_like(q)),
        )
