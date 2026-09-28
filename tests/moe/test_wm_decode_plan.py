"""CPU mirror checks for the weight-major NVFP4 MoE decode (``wm``) plans.

The kernel's stage copies, F8_128x4 scale-word addressing and in-CTA routing are
mirrored here literally, so a mistake in that arithmetic fails without a GPU.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest
import torch

from b12x.moe._shared.kernels.wm_geometry import (
    FC1_ROWS,
    FC2_ROWS,
    PASS_ROWS,
    THREADS,
    wm_geometry,
    wm_scale_offset,
    wm_stage_plan,
)

QWEN = dict(hidden_size=2560, intermediate_size=320, num_experts=512, top_k=10)


def _plane_offsets(rows: int, cols: int) -> torch.Tensor:
    """Byte offset of logical scale (row, col) in a b12x/FlashInfer F8_128x4 plane."""
    logical = torch.arange(rows * cols).reshape(1, rows, cols)
    stored = logical.reshape(1, rows // 128, 4, 32, cols // 4, 4)
    stored = stored.permute(0, 1, 4, 3, 2, 5).reshape(-1)
    pos = torch.empty_like(stored)
    pos[stored] = torch.arange(stored.numel())
    return pos.reshape(rows, cols)


@pytest.mark.parametrize("rows,cols", [(640, 160), (2560, 20)])
def test_scale_word_offsets_match_f8_128x4_plane(rows, cols):
    pos = _plane_offsets(rows, cols)
    r = torch.arange(rows)[:, None]
    k64 = torch.arange(cols // 4)[None, :]
    ours = wm_scale_offset(r, k64, atom_bytes=(cols // 4) * 512)
    # A K64 word holds the four consecutive K16 scales of its row, low byte first.
    for b in range(4):
        assert torch.equal(ours + b, pos[:, b::4])


def test_scale_word_offsets_match_b12x_swizzle():
    swizzle = pytest.importorskip("b12x._lib.intrinsics").swizzle_block_scale
    rows, cols = 640, 160
    logical = torch.arange(rows * cols, dtype=torch.int64).reshape(rows, cols)
    stored = swizzle(logical).reshape(-1)
    ours = wm_scale_offset(torch.arange(rows)[:, None], torch.arange(cols // 4)[None, :],
                           atom_bytes=(cols // 4) * 512)
    assert torch.equal(stored[ours], logical[:, ::4])


def _mirror_stage_copies(g: dict, stage: int):
    """Global chunk/word offsets one item stage copies, exactly as the kernel issues them."""
    kind, row0, kc, _ = wm_stage_plan(g, stage)
    payload, words = [], []
    if kind == "fc1":
        per_row = g["k2"] // 16
        for tid in range(THREADS):
            for i in range(g["fc1_chunks"] // THREADS):
                idx = tid + i * THREADS
                r, v = divmod(idx, per_row)
                payload.append((row0 * g["k2"] + idx * 16, r * g["p1"] + v * 16))
            for i in range(-(-g["fc1_words"] // THREADS)):
                idx = tid + i * THREADS
                if idx < g["fc1_words"]:
                    r, j = divmod(idx, g["fc1_k64"])
                    words.append(wm_scale_offset(row0 + r, kc * g["fc1_k64"] + j,
                                                 atom_bytes=g["sf13_atom"]))
    else:
        per_row = g["i2"] // 16
        for tid in range(THREADS):
            for i in range(-(-g["fc2_chunks"] // THREADS)):
                idx = tid + i * THREADS
                if idx < g["fc2_chunks"]:
                    r, v = divmod(idx, per_row)
                    payload.append((row0 * g["i2"] + idx * 16, r * g["p2"] + v * 16))
            for i in range(-(-g["fc2_words"] // THREADS)):
                idx = tid + i * THREADS
                if idx < g["fc2_words"]:
                    r, j = divmod(idx, g["fc2_k64"])
                    words.append(wm_scale_offset(row0 + r, j, atom_bytes=g["sf2_atom"]))
    return kind, payload, words


@pytest.mark.parametrize("max_tokens", [1, 5, 16, 20, 32])
def test_item_stages_stream_every_expert_byte_once(max_tokens):
    g = wm_geometry(**QWEN, max_tokens=max_tokens)
    K, I = g["K"], g["I"]
    counts = {
        "fc1": Counter(), "fc2": Counter(), "fc1_w": Counter(), "fc2_w": Counter(),
    }
    for stage in range(g["spi"]):
        kind, payload, words = _mirror_stage_copies(g, stage)
        dsts = [d for _, d in payload]
        assert len(set(dsts)) == len(dsts) and max(dsts) + 16 <= g["slot_payload"]
        counts[kind].update(src for src, _ in payload)
        counts[kind + "_w"].update(words)
    assert sorted(counts["fc1"]) == list(range(0, 2 * I * g["k2"], 16))
    assert sorted(counts["fc2"]) == list(range(0, K * g["i2"], 16))
    assert sorted(counts["fc1_w"]) == list(range(0, g["sf13_expert"], 4))
    assert sorted(counts["fc2_w"]) == list(range(0, g["sf2_expert"], 4))
    for c in counts.values():
        assert set(c.values()) == {1}
    # FC1 stages pair 32 up rows with the same 32 channels of gate rows.
    for stage in range(g["fc1_stages"]):
        _, row0, _, cb = wm_stage_plan(g, stage)
        assert row0 % I == cb * FC1_ROWS
    assert g["spi"] == g["fc1_stages"] + K // FC2_ROWS


def _mirror_items(ids: list[int], num_experts: int):
    """In-CTA routing: bitmap ranks, second-pass map, and the item -> expert walk."""
    words = [0] * (num_experts // 32)
    big = [0] * (num_experts // 32)
    cnt = Counter(e for e in ids if 0 <= e < num_experts)
    for e in cnt:
        words[e >> 5] |= 1 << (e & 31)
        if cnt[e] > PASS_ROWS:
            big[e >> 5] |= 1 << (e & 31)

    def prefix(bm):
        out, run = [], 0
        for w in bm:
            out.append(run)
            run += bin(w).count("1")
        return out + [run]

    pre, bpre = prefix(words), prefix(big)
    d = pre[-1]

    def item_expert(item):
        bm, pr, r = (words, pre, item) if item < d else (big, bpre, item - d)
        word = 0
        for w in range(1, len(bm)):
            if pr[w] <= r:
                word = w
        bits, k = bm[word], r - pr[word]
        while k > 0:
            bits &= bits - 1
            k -= 1
        return word * 32 + ((bits & -bits) - 1).bit_count()

    items = []
    for item in range(d + bpre[-1]):
        e = item_expert(item)
        p = 0 if item < d else 1
        hits = [i for i, x in enumerate(ids) if x == e]
        items.append((e, p, hits[PASS_ROWS * p: PASS_ROWS * (p + 1)]))
    return items


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("m", [1, 5, 13, 20, 32])
def test_items_cover_every_route_once(seed, m):
    rng = random.Random(seed * 101 + m)
    E, topk = 512, 10
    pattern = seed % 3
    ids = []
    for t in range(m):
        if pattern == 0:
            row = rng.sample(range(E), topk)           # spread
        elif pattern == 1:
            row = rng.sample(range(24), topk)          # clustered, shared experts
        else:
            row = [7] + rng.sample(range(8, E), topk - 1)  # one hot expert: 2 passes
        if t == m - 1 and m > 1:
            row[-1] = -1                               # masked route
        ids.extend(row)
    items = _mirror_items(ids, E)
    experts = [e for e, p, _ in items if p == 0]
    assert experts == sorted(set(e for e in ids if e >= 0))
    covered = Counter(i for _, _, rows in items for i in rows)
    assert set(covered.values()) <= {1}
    assert sorted(covered) == [i for i, e in enumerate(ids) if e >= 0]
    assert all(1 <= len(rows) <= PASS_ROWS for _, _, rows in items)
    for grid in (1, 7, 48):
        owned = [i for b in range(grid) for i in range(b, len(items), grid)]
        assert sorted(owned) == list(range(len(items)))


def test_geometry_fits_shared_memory_and_rejects_unsupported_shapes():
    for m in (1, 16, 17, 32):
        g = wm_geometry(**QWEN, max_tokens=m)
        assert g["smem_bytes"] <= 101376
        assert g["passes"] == (1 if m <= 16 else 2)
    with pytest.raises(ValueError):
        wm_geometry(**QWEN, max_tokens=33)
    with pytest.raises(ValueError):
        wm_geometry(hidden_size=4096, intermediate_size=320, num_experts=512, top_k=10,
                    max_tokens=5)
