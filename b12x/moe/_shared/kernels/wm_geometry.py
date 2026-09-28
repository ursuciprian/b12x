"""Static geometry of the weight-major NVFP4 MoE decode kernel (``wm_decode``).

Pure Python and free of CUDA imports, so planning code and the CPU mirror tests
can use the exact stage, scale-plane and shared-memory arithmetic the kernel
compiles against.
"""

from __future__ import annotations

# Fixed tiling (see the module docstring). The pass width is the QMMA M16 atom.
PASS_ROWS = 16
FC1_ROWS = 8  # one FC1 stage = 8 full contiguous w13 rows (LPDDR row locality)
FC2_ROWS = 64
NUM_WARPS = 4
THREADS = NUM_WARPS * 32
STAGES = 5
MAX_SMEM_BYTES = 101376  # SM120/SM121 opt-in per-block limit


def wm_geometry(*, hidden_size: int, intermediate_size: int, num_experts: int,
                top_k: int, max_tokens: int) -> dict:
    """Static geometry shared by the kernel, the launcher and the CPU mirror tests."""
    K, I, E = int(hidden_size), int(intermediate_size), int(num_experts)
    if K % (64 * NUM_WARPS) or K % 128 or K % FC2_ROWS:
        raise ValueError(f"wm decode needs hidden_size % {64 * NUM_WARPS} == 0, got {K}")
    if I % 64 or I % FC1_ROWS:
        raise ValueError(f"wm decode needs intermediate_size % 64 == 0, got {I}")
    if E % 32 or E > 1024:
        raise ValueError(f"wm decode needs num_experts % 32 == 0 and <= 1024, got {E}")
    if not 1 <= max_tokens <= 2 * PASS_ROWS:
        raise ValueError(f"wm decode supports 1..{2 * PASS_ROWS} tokens, got {max_tokens}")
    g = dict(K=K, I=I, E=E, top_k=int(top_k), max_tokens=int(max_tokens))
    g["passes"] = -(-max_tokens // PASS_ROWS)
    g["k2"] = K // 2                      # w13 row bytes
    g["i2"] = I // 2                      # down row bytes
    g["kchunks"] = 1                      # FC1 stages per row group (full K per stage)
    g["fc1_k64"] = K // 64                # K64 slices per FC1 stage
    g["warp_k64"] = K // 64 // NUM_WARPS  # FC1 split-K: K64 slices per warp
    g["ch_blocks"] = I // FC1_ROWS
    g["fc1_stages"] = g["ch_blocks"] * 2 * g["kchunks"]
    g["fc2_k64"] = I // 64
    g["fc2_stages"] = K // FC2_ROWS
    g["spi"] = g["fc1_stages"] + g["fc2_stages"]  # stages per item
    # Shared row strides are padded by 16 B so the QMMA fragment loads of the eight
    # q rows of a warp land in distinct banks.
    g["p1"] = K // 2 + 16
    g["p2"] = g["i2"] + 16
    g["fc1_chunks"] = FC1_ROWS * (K // 2) // 16
    g["fc2_chunks"] = FC2_ROWS * g["i2"] // 16
    g["fc1_words"] = FC1_ROWS * g["fc1_k64"]
    g["fc2_words"] = FC2_ROWS * g["fc2_k64"]
    g["slot_payload"] = max(FC1_ROWS * g["p1"], FC2_ROWS * g["p2"])
    g["slot_bytes"] = g["slot_payload"] + 4 * max(g["fc1_words"], g["fc2_words"])
    # F8_128x4 planes: a 128-row atom holds (cols/4) 512-byte K64 groups.
    g["sf13_atom"] = (K // 64) * 512
    g["sf13_expert"] = -(-(2 * I) // 128) * g["sf13_atom"]
    g["sf2_atom"] = (I // 64) * 512
    g["sf2_expert"] = -(-K // 128) * g["sf2_atom"]
    g["a_stride"] = g["k2"] + 16
    g["as_words"] = K // 64 + 1           # odd word stride: conflict-free SFA reads
    g["hq_stride"] = g["i2"] + 16
    g["hs_stride"] = 4 * (I // 64)
    g["routes"] = max_tokens * top_k
    g["route_chunks"] = -(-g["routes"] // 32)
    g["bitmap_words"] = E // 32
    off = 0
    for name, size in (
        ("ring", STAGES * g["slot_bytes"]),
        ("a", PASS_ROWS * g["a_stride"]),
        ("as", PASS_ROWS * 4 * g["as_words"]),
        ("h", PASS_ROWS * I * 2),
        # Routing counts are dead after the prologue; the FC1 split-K partials reuse them.
        ("cnt", max(4 * E if g["passes"] > 1 else 16, 2 * NUM_WARPS * PASS_ROWS * FC1_ROWS * 4)),
        ("bm", 4 * g["bitmap_words"]),
        ("big", 4 * g["bitmap_words"]),
        ("pre", 4 * (g["bitmap_words"] + 4)),
        ("bpre", 4 * (g["bitmap_words"] + 4)),
        ("tok", 4 * PASS_ROWS),
        ("wgt", 4 * PASS_ROWS),
        ("nrows", 16),
    ):
        g[f"off_{name}"] = off
        off += -(-size // 16) * 16
    g["smem_bytes"] = off
    g["off_part"] = g["off_cnt"]
    if g["hq_stride"] * PASS_ROWS > PASS_ROWS * g["a_stride"] or \
            g["hs_stride"] * PASS_ROWS > PASS_ROWS * 4 * g["as_words"]:
        raise ValueError("wm decode intermediate does not fit the aliased A region")
    if off > MAX_SMEM_BYTES:
        raise ValueError(f"wm decode needs {off} B of shared memory (> {MAX_SMEM_BYTES})")
    if g["fc1_chunks"] % THREADS or g["fc2_chunks"] > g["fc1_chunks"]:
        raise ValueError("wm decode stage copy geometry is unsupported")
    return g


def wm_stage_plan(g: dict, stage: int) -> tuple[str, int, int, int]:
    """Stage -> (kind, first weight row, K chunk, channel block); the CPU mirror of the kernel."""
    if stage < g["fc1_stages"]:
        cb = stage // (2 * g["kchunks"])
        half = (stage // g["kchunks"]) % 2
        kc = stage % g["kchunks"]
        return "fc1", half * g["I"] + cb * FC1_ROWS, kc, cb
    rb = stage - g["fc1_stages"]
    return "fc2", rb * FC2_ROWS, 0, rb


def wm_scale_offset(row, k64, *, atom_bytes: int):
    """Byte offset of the 4-scale K64 word of ``row`` in one expert's F8_128x4 plane."""
    return (row >> 7) * atom_bytes + k64 * 512 + (row & 31) * 16 + ((row & 127) >> 5) * 4
