"""Correctness oracle for deferred GDN checkpoints; NOT a serving implementation.

Run: python3 validation/gdn/deferred_checkpoints.py
Keep the pre-verification state immutable, record the exact FP32 decay and
rank-one update at each step, then replay only the accepted prefix. This avoids
storing every full state. A CuTe implementation must preserve the same FP32
multiply/add contraction and checkpoint lifetime before it can replace GDN.
"""
import torch


def verify_with_records(initial, keys, values, decay, beta):
    state = initial.clone()
    deltas, checkpoints = [], []
    ratio = state.shape[0] // keys.shape[1]
    for key, value, d, b in zip(keys, values, decay, beta):
        expanded_key = key.repeat_interleave(ratio, dim=0)
        state = state * d[:, None, None]
        prediction = (state * expanded_key[:, None, :]).sum(-1)
        delta = (value - prediction) * b[:, None]
        state = torch.addcmul(state, delta[:, :, None], expanded_key[:, None, :])
        deltas.append(delta)
        # Reference-only storage for checking every possible acceptance count.
        checkpoints.append(state.clone())
    return checkpoints, torch.stack(deltas)


def commit_accepted(initial, keys, decay, deltas, accepted):
    if not 0 <= accepted <= len(keys):
        raise ValueError("accepted count must select a verified prefix")
    state = initial.clone()
    ratio = state.shape[0] // keys.shape[1]
    for step in range(accepted):
        state = state * decay[step, :, None, None]
        state = torch.addcmul(
            state, deltas[step, :, :, None],
            keys[step].repeat_interleave(ratio, dim=0)[:, None, :],
        )
    return state


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20260922)
    checks = 0
    for tokens in (1, 2, 5, 8):
        for scale in (0.01, 1.0, 10.0):
            initial = torch.randn(24, 128, 128) * scale
            keys = torch.nn.functional.normalize(torch.randn(tokens, 8, 128), dim=-1)
            values = torch.randn(tokens, 24, 128)
            decay = torch.rand(tokens, 24)
            beta = torch.rand(tokens, 24).bfloat16().float()
            saved_initial = initial.clone()
            checkpoints, deltas = verify_with_records(initial, keys, values, decay, beta)
            for accepted in range(tokens + 1):
                expected = initial if accepted == 0 else checkpoints[accepted - 1]
                actual = commit_accepted(initial, keys, decay, deltas, accepted)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                checks += 1
            torch.testing.assert_close(initial, saved_initial, rtol=0, atol=0)
    # Per rank, 36 layers, 24 value heads, 8 key heads, FP32 recurrent state.
    snapshot = 36 * 24 * 128 * 128 * 4
    records = 36 * 5 * (24 * 128 + 8 * 128 + 24) * 4
    old_bytes = 6 * snapshot  # one read plus five full checkpoint writes
    new_bytes = 3 * snapshot + 2 * records  # verify read, commit read/write, records
    print(f"PASS: {checks} accepted-prefix states are bit-identical in the FP32 oracle")
    print(f"c8 state traffic model: {8 * old_bytes / 2**30:.3f} -> {8 * new_bytes / 2**30:.3f} GiB/rank/step")
    print("GPU implementation, allocator lifetime, prefix-cache interoperability and E2E speed remain unvalidated.")


if __name__ == "__main__":
    main()
