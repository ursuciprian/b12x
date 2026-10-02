"""CPU checks of the flint decode split (no GPU): ranges tile the work and stay balanced."""

from __future__ import annotations

import pytest

from b12x.moe._shared.kernels.wm_geometry import flint_geometry, flint_ranges, wm_geometry


@pytest.mark.parametrize("intermediate", [320, 640])
@pytest.mark.parametrize("max_tokens", [5, 10, 20, 32])
def test_flint_geometry_fits(intermediate, max_tokens):
    g = flint_geometry(wm_geometry(hidden_size=2560, intermediate_size=intermediate,
                                   num_experts=512, top_k=10, max_tokens=max_tokens))
    assert g["off_route"] >= g["off_nrows"] + 16
    assert g["fc1_units"] * 2 == g["fc1_stages"]
    # flint's h scratch lives in the dynamic packed_input (rows >= routes, K/2 bytes each).
    assert 2 * g["I"] <= g["K"] // 2


@pytest.mark.parametrize("items", [1, 10, 33, 60, 96, 151, 330])
@pytest.mark.parametrize("grid", [16, 48])
def test_flint_ranges_tile_and_balance(items, grid):
    units = 80
    spans = [flint_ranges(items, units, grid, b) for b in range(grid)]
    assert spans[0][0] == 0 and spans[-1][1] == items * units
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    sizes = [hi - lo for lo, hi in spans]
    assert max(sizes) - min(sizes) <= 1


def test_flint_schedule_controls(monkeypatch):
    """Unset: wm controls unchanged (same keys); flint: the schedule joins the query."""
    from b12x.moe.fused_moe import _impl

    monkeypatch.setenv("B12X_MOE_DECODE_BACKEND", "wm")
    monkeypatch.delenv("B12X_MOE_WM_SCHEDULE", raising=False)
    assert "wm_schedule" not in _impl._wm_decode_controls()
    monkeypatch.setenv("B12X_MOE_WM_SCHEDULE", "flint")
    assert _impl._wm_decode_controls()["wm_schedule"] == "flint"
    monkeypatch.setenv("B12X_MOE_WM_SCHEDULE", "bogus")
    with pytest.raises(ValueError):
        _impl._wm_decode_controls()
