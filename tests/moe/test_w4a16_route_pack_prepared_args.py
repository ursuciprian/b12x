"""Host-only: prepared W4A16 route-pack launches pass the full Triton signature.

Compiled Triton launchers take every parameter, constexprs included. Omitting
them (b12x before upstream 8783519a) shifts the argument list the first time a
prepared W4A16 route pack runs, which the A16 token cutoff made reachable for
ModelOpt NVFP4 MoE (r7-a16-8 boot died in moe.decode preparation).
"""

import pytest
import torch

import b12x.moe._shared.kernels.w4a16.route_pack as rp

KERNELS = {
    "small_prefix": rp._pack_topk_routes_small_prefix_kernel,
    "count": rp._w4a16_route_count_kernel,
    "prefix": rp._w4a16_route_prefix_from_counts_kernel,
    "post_prefix": rp._pack_topk_routes_post_prefix_kernel,
    "sort": rp._pack_topk_routes_sort_kernel,
}


@pytest.mark.parametrize("tokens", (4, 4096))
@pytest.mark.parametrize("mapped", (False, True))
def test_prepared_launch_arity_matches_kernel_signature(monkeypatch, tokens, mapped):
    topk, block, experts = 2, 8, 32
    numel_capacity, routes, blocks = rp.route_pack_capacity(tokens * topk, block, experts, topk=topk)
    block_e = rp._next_power_of_2(experts)
    small = (
        rp._next_power_of_2(max(routes, 1)) <= rp._SMALL_PREFIX_MAX_PACKED_ROUTES
        and rp._next_power_of_2(max(blocks, 1)) <= rp._SMALL_PREFIX_MAX_ROUTE_BLOCKS
    )
    names = ("small_prefix", "sort") if small else ("count", "prefix", "post_prefix", "sort")
    programs = {}
    for name in names:
        for dt in (torch.int32, torch.int64):
            for m in (False, True):
                programs[(name, dt, m)] = name
    launches = rp.W4A16RoutePackLaunches(
        numel_capacity=int(numel_capacity), block_size=block, num_experts=experts,
        max_packed_routes=max(routes, 1), max_route_blocks=max(blocks, 1),
        use_small_prefix=small, programs=programs,
    )
    calls = []
    monkeypatch.setattr(rp, "_launch_prepared", lambda program, grid, *args: calls.append((program, args)))
    ids = torch.randint(0, experts, (tokens, topk), dtype=torch.int32)
    expert_map = torch.arange(experts, dtype=torch.int32) if mapped else None
    rp.pack_topk_routes_by_expert(
        ids, block, experts, expert_map=expert_map, launches=launches,
        packed_route_indices=torch.empty(launches.max_packed_routes, dtype=torch.int32),
        block_expert_ids=torch.empty(launches.max_route_blocks, dtype=torch.int32),
        packed_route_count=torch.empty(1, dtype=torch.int32),
        expert_offsets=torch.empty(experts + 1, dtype=torch.int32),
        expert_counts=torch.empty(experts, dtype=torch.int32),
    )
    assert {name for name, _ in calls} == set(names)
    for name, args in calls:
        assert len(args) == len(KERNELS[name].arg_names), name
