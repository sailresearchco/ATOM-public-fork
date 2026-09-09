"""Prepare KDA chunk metadata from scheduler-owned host lengths."""

import torch


def prepare_kda_seq_bounds(seqlens_cpu):
    """CPU sequence bounds for FlashKDA's segmentation decision."""
    if isinstance(seqlens_cpu, torch.Tensor):
        assert seqlens_cpu.device.type == "cpu"
    bounds, offset = [], 0
    for length in seqlens_cpu:
        length = int(length)
        assert length >= 0
        bounds.append((offset, offset + length))
        offset += length
    return tuple(bounds)


def prepare_kda_chunk_indices(seqlens_cpu, *, device, dtype=torch.int32):
    """Keep both AITER chunk sizes ready without reading GPU offsets back.

    AITER selects 32 or 64 after checking its FlashKDA eligibility. Preparing
    both preserves that selection and shares the indices across KDA layers.
    """
    if isinstance(seqlens_cpu, torch.Tensor):
        assert seqlens_cpu.device.type == "cpu"
    lengths = [int(n) for n in seqlens_cpu]
    assert all(n >= 0 for n in lengths)
    result = {}
    for chunk_size in (32, 64):
        rows = [
            (sequence, chunk)
            for sequence, length in enumerate(lengths)
            for chunk in range((length + chunk_size - 1) // chunk_size)
        ]
        host = torch.tensor(rows, dtype=dtype, device="cpu").reshape(-1, 2)
        if torch.device(device).type == "cuda":
            host = host.pin_memory()
        result[chunk_size] = host.to(device=device, non_blocking=True)
    return result
