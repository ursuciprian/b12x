"""Host-only selection rules for prepared fused-MoE variants and W4A16 launches."""

import pytest
import torch

from b12x.moe.fused_moe._preparation import (
    _FusedMoeCapacityState,
    _W4A16PrimaryLaunches,
    variant_for,
)


class _Variant:
    def __init__(self, count):
        self.count = count

    def bind(self, **kwargs):
        return (self.count, kwargs)


def test_variant_for_preserves_exact_counts_and_reuses_prefill_capacity():
    variants = {count: _Variant(count) for count in (1, 2, 4, 8, 125, 128)}
    assert variant_for(variants, 4) is variants[4]
    assert variant_for(variants, 125) is variants[125]
    for count in (3, 11, 31, 126, 128):
        assert variant_for(variants, count) is variants[128]
    with pytest.raises(ValueError, match="exceeds prepared MoE capacity 128"):
        variant_for(variants, 129)


def test_capacity_state_binds_the_planned_variant_with_the_live_activations():
    variants = {count: _Variant(count) for count in (4, 128)}
    state = _FusedMoeCapacityState(variants)
    activations = torch.empty(11, 16)
    count, kwargs = state.bind(a=activations, topk_ids=None)
    assert count == 128
    assert kwargs["a"] is activations
    with pytest.raises(TypeError):
        state.bind(a=torch.empty(11))
    with pytest.raises(ValueError, match="exceeds prepared MoE capacity 128"):
        state.bind(a=torch.empty(129, 16))


def test_a16_capacity_covers_unlisted_counts_without_crossing_cutoff():
    variants = {count: _Variant(count) for count in (4, 32, 64, 128)}
    state = _FusedMoeCapacityState(variants, a16_max_tokens=32)
    for rows, expected in ((1, 32), (4, 4), (17, 32), (31, 32), (32, 32),
                           (33, 128), (64, 64), (127, 128)):
        count, _ = state.bind(a=torch.empty(rows, 16))
        assert count == expected
    for rows in (0, 129):
        with pytest.raises(ValueError):
            state.bind(a=torch.empty(rows, 16))


def _launches(*, direct, route_pack, route_mode="auto"):
    return _W4A16PrimaryLaunches(
        tokens=8, route_mode=route_mode, packed="packed", packed_mapped="packed_mapped",
        direct="direct" if direct else None,
        direct_mapped="direct_mapped" if direct else None,
        topk_sum="topk_sum", mapped_topk_sum="mapped_topk_sum",
        route_pack=route_pack,
    )


@pytest.mark.parametrize("route_mode", ("auto", "direct"))
def test_w4a16_select_preserves_exact_direct_and_dynamic_packed_routes(route_mode):
    launches = _launches(direct=True, route_pack="route_pack", route_mode=route_mode)
    assert launches.select(
        tokens=8, route_ids_dtype=torch.int32, has_route_map=False, activation_amax=None,
    ) == ("direct", "topk_sum", None)
    assert launches.select(
        tokens=8, route_ids_dtype=torch.int64, has_route_map=False, activation_amax=None,
    ) == ("packed", "topk_sum", "route_pack")
    assert launches.select(
        tokens=3, route_ids_dtype=torch.int32, has_route_map=False, activation_amax=None,
    ) == ("packed", "topk_sum", "route_pack")
    assert launches.select(
        tokens=3, route_ids_dtype=torch.int64, has_route_map=True, activation_amax=None,
    ) == ("packed_mapped", "topk_sum", "route_pack")
    with pytest.raises(RuntimeError, match="requested=9, prepared=8"):
        launches.select(
            tokens=9, route_ids_dtype=torch.int32, has_route_map=False, activation_amax=None,
        )
