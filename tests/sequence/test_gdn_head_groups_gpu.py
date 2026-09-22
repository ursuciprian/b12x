"""One planned callable must handle changing live batches under graph replay."""
import gc
import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.sequence import gdn_decode as gdn
from ..conftest import require_b12x as require_sm120
from .test_gdn_decode import _make_case, _prepare, _reference, _prepared_case_lifetime  # noqa: F401


@pytest.mark.parametrize("head_group_size", [1, 2, 3])
def test_compact_groups_match_oracle_with_frozen_resolution(head_group_size):
    device = require_sm120()
    binding, tensors = _make_case(
        device=device, query_lengths=(5,) * 16, max_seqs=16,
        max_tokens=80, columns=5, state_slots=81,
        head_group_size=head_group_size,
    )
    legacy = _prepare(binding._state.caps, tensors, override=gdn.GdnConfig(
        backend="cutedsl", recurrent_block_v=32,
    ))
    initial_state = binding.recurrent_state.clone()
    gdn.run(binding)
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard("GDN grouping live-count coverage"):
        with torch.cuda.graph(graph):
            gdn.run(binding)
        addresses = (binding.output.data_ptr(), binding.recurrent_state.data_ptr())
        cases = [(5,) * c for c in (1, 4, 8, 10, 16)] + [(5, 2, 0, 1, 4), ()]
        for lengths in cases:
            binding.query_start_loc.fill_(sum(lengths))
            binding.query_start_loc[0] = 0
            if lengths:
                binding.query_start_loc[1:len(lengths) + 1].copy_(
                    torch.tensor(lengths, device=device, dtype=torch.int32).cumsum(0)
                )
            binding.num_seqs.fill_(len(lengths))
            binding.num_tokens.fill_(sum(lengths))
            binding.num_accepted_tokens.copy_(
                torch.arange(16, device=device, dtype=torch.int32).remainder(5).add(1)
            )
            expected_state = initial_state.clone()
            expected_output = _reference(binding, state=expected_state)
            binding.recurrent_state.copy_(initial_state)
            gdn.run(legacy)
            legacy_output = binding.output.clone()
            legacy_state = binding.recurrent_state.clone()
            binding.recurrent_state.copy_(initial_state)
            binding.output.fill_(float("nan"))
            gc.collect()
            torch.cuda.synchronize(device)
            allocated = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_allocated(device) == allocated
            assert torch.cuda.max_memory_allocated(device) == allocated
            assert addresses == (binding.output.data_ptr(), binding.recurrent_state.data_ptr())
            torch.testing.assert_close(binding.output, expected_output, rtol=1e-2, atol=2e-2)
            torch.testing.assert_close(binding.recurrent_state, expected_state, rtol=1e-5, atol=2e-5)
            torch.testing.assert_close(binding.output, legacy_output, rtol=0, atol=0)
            torch.testing.assert_close(binding.recurrent_state, legacy_state, rtol=0, atol=0)
