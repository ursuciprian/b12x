"""CPU checks of the flint decode split (no GPU): ranges tile the work and stay balanced."""

from __future__ import annotations

import pytest

from b12x.moe._shared.kernels.wm_geometry import (
    FLINT_MAX_PASSES,
    MAX_SMEM_BYTES,
    PASS_ROWS,
    flint_geometry,
    flint_ranges,
    wm_geometry,
)

QWEN = dict(hidden_size=2560, num_experts=512, top_k=10)


def _flint(intermediate, max_tokens):
    return flint_geometry(wm_geometry(**QWEN, intermediate_size=intermediate,
                                      max_tokens=max_tokens, max_passes=FLINT_MAX_PASSES))


@pytest.mark.parametrize("intermediate", [320, 640])
@pytest.mark.parametrize("max_tokens", [5, 10, 20, 32, 33, 40, 48, 64])
def test_flint_geometry_fits(intermediate, max_tokens):
    g = _flint(intermediate, max_tokens)
    assert g["off_route"] >= g["off_nrows"] + 16
    assert g["fc1_units"] * 2 == g["fc1_stages"]
    assert g["passes"] == -(-max_tokens // PASS_ROWS)
    # One CTA per SM (cooperative launch, grid <= SMs) needs the opt-in per-block limit.
    assert g["smem_bytes"] <= MAX_SMEM_BYTES
    # Same ring and FC2 stage as wm at this width: the pass count only adds bitmaps.
    wm = wm_geometry(**QWEN, intermediate_size=intermediate, max_tokens=min(max_tokens, 32))
    assert (g["fc2_rows"], g["stages"]) == (wm["fc2_rows"], wm["stages"])
    # Per-pass bitmaps and prefix tables do not overlap and stay inside the layout.
    assert g["off_bm"] + g["passes"] * g["bm_stride"] <= g["off_pre"]
    assert g["off_pre"] + g["passes"] * g["pre_stride"] <= g["off_tok"]
    # flint's h scratch lives in the dynamic packed_input (rows >= routes, K/2 bytes each).
    assert 2 * g["I"] <= g["K"] // 2


def test_token_limits():
    """wm keeps 32 tokens; flint takes up to 64."""
    with pytest.raises(ValueError):
        wm_geometry(**QWEN, intermediate_size=640, max_tokens=33)
    assert _flint(640, 64)["passes"] == 4
    with pytest.raises(ValueError):
        _flint(640, 65)


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


def test_flint_max_tokens_control(monkeypatch):
    """The cap stays 32 by default; B12X_MOE_WM_MAX_TOKENS may go to 64 only under flint."""
    from b12x.moe.fused_moe import _impl

    monkeypatch.setenv("B12X_MOE_DECODE_BACKEND", "wm")
    monkeypatch.delenv("B12X_MOE_WM_MAX_TOKENS", raising=False)
    for schedule in ("item", "flint"):
        monkeypatch.setenv("B12X_MOE_WM_SCHEDULE", schedule)
        assert _impl._wm_decode_controls()["wm_max_tokens"] == 32
    monkeypatch.setenv("B12X_MOE_WM_MAX_TOKENS", "64")
    assert _impl._wm_decode_controls()["wm_max_tokens"] == 64
    monkeypatch.setenv("B12X_MOE_WM_MAX_TOKENS", "65")
    with pytest.raises(ValueError):
        _impl._wm_decode_controls()
    monkeypatch.setenv("B12X_MOE_WM_MAX_TOKENS", "33")
    monkeypatch.setenv("B12X_MOE_WM_SCHEDULE", "item")
    with pytest.raises(ValueError):
        _impl._wm_decode_controls()
