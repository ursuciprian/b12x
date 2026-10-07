"""CPU check of the MTP draft-reuse selection layout against the sparse-GQA reader (no GPU).

The real ``_prepare_kernel`` runs in the Triton interpreter. The reader is the address math of
``paged/_selected_forward.py`` ``_mma_selected_tile``: row r, column c reads flat offset
``r * selection_width + c`` for ``c < selection_width`` and keeps ``0 <= position <= query_position``.
Run alone (the interpreter is chosen when the kernel module is first imported):
    TRITON_INTERPRET=1 python3 -m pytest -q tests/attention/test_qsa_draft_reuse_stride_cpu.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

WIDTH, TAIL = 2051, 4
# k67a: the reuse attention launches programs.sparse, compiled for caps.selection_width columns
# (_contract.py), and the prepare kernel writes rows at that stride without the chain tail.
READ_WIDTH, ATTENDS_TAIL = WIDTH, False


def _reads(flat: torch.Tensor, row: int, width: int, query_position: int) -> list[int]:
    """Logical positions the selected-position kernel attends for one row."""
    got = flat[row * width : row * width + width].tolist()
    return [p for p in got if 0 <= p <= query_position]


def _requests():
    """Two requests at long context (full 2051-column anchors ending in the newest tokens), step 3."""
    g = torch.Generator().manual_seed(113)
    out = []
    for anchor in (5000, 6000):
        older = torch.randperm(anchor - 2, generator=g)[: WIDTH - 3].sort().values
        sel = torch.cat((older, torch.arange(anchor - 2, anchor + 1))).to(torch.int32)
        out.append((anchor, anchor + 2, sel))  # (anchor, query position = anchor + step - 1, selection)
    return out


def test_served_layout_reads_row1_shifted_by_four_columns():
    """21e0b201: rows written at a 2055 stride, read at 2051 -> row 1 is read 4 columns early."""
    reqs = _requests()
    buf = torch.full((2, WIDTH + TAIL), -1, dtype=torch.int32)
    for r, (anchor, pos, sel) in enumerate(reqs):
        buf[r, :WIDTH] = sel
        chain = torch.arange(anchor + 1, anchor + 1 + TAIL)
        buf[r, WIDTH:] = torch.where(chain <= pos, chain, -1).to(torch.int32)
    flat = buf.reshape(-1)
    anchor0, pos0, _ = reqs[0]
    _, pos1, sel1 = reqs[1]
    assert _reads(flat, 0, WIDTH, pos0) == reqs[0][2].tolist()  # row 0 is fine
    row1 = _reads(flat, 1, WIDTH, pos1)
    assert row1[:2] == [anchor0 + 1, anchor0 + 2]  # request 0's chain, read as request 1 positions
    assert row1[2:] == sel1[: WIDTH - 4].tolist()
    lost = set(sel1.tolist()) - set(row1)
    assert {5998, 5999, 6000} <= lost  # request 1's newest prefix tokens are dropped


@pytest.mark.skipif(
    "b12x.attention.qsa._draft_selection" in sys.modules and os.environ.get("TRITON_INTERPRET") != "1",
    reason="kernel module already imported without the Triton interpreter",
)
def test_prepare_kernel_layout_matches_reader():
    import triton

    from b12x.attention.qsa._draft_selection import _prepare_kernel

    reqs = _requests()
    max_requests = 4
    src_pos = torch.tensor([a for a, _, _ in reqs], dtype=torch.int64)
    src_sel = torch.stack([s for _, _, s in reqs]).contiguous()
    source_rows = torch.tensor([0, 1, -1, -1], dtype=torch.int64)
    num_src = torch.tensor([2], dtype=torch.int32)
    request_ids = torch.tensor([0, 1], dtype=torch.int32)
    positions = torch.tensor([p for _, p, _ in reqs], dtype=torch.int64)
    # the scratch region reserves WIDTH + TAIL per row; -7 marks bytes the kernel never wrote
    flat = torch.full((max_requests * (WIDTH + TAIL),), -7, dtype=torch.int32)
    _prepare_kernel[(2,)](
        src_pos, src_sel, source_rows, num_src, request_ids, positions, flat,
        2, 2, max_requests, WIDTH=WIDTH, TAIL=TAIL, BLOCK=triton.next_power_of_2(WIDTH + TAIL),
    )
    for r, (anchor, pos, sel) in enumerate(reqs):
        expected = sel.tolist() + (list(range(anchor + 1, pos + 1)) if ATTENDS_TAIL else [])
        got = _reads(flat, r, READ_WIDTH, pos)
        assert got == expected, f"row {r}: {len(got)} read vs {len(expected)} expected"
        assert -7 not in flat[r * READ_WIDTH : (r + 1) * READ_WIDTH].tolist()
