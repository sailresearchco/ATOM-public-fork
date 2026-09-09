"""KDA metadata must not synchronize ranks before cross-node collectives."""
import importlib
from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.kimi_k3.prefill_metadata import (
    prepare_kda_chunk_indices, prepare_kda_seq_bounds,
)


@pytest.mark.parametrize("lengths", [[], [0, 1, 0, 65], [6, 2304], [2304, 5888]])
def test_chunk_indices_from_host_lengths(lengths):
    result = prepare_kda_chunk_indices(lengths, device="cpu")
    bounds = prepare_kda_seq_bounds(lengths)
    assert [end - start for start, end in bounds] == lengths
    assert all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1))
    for size, indices in result.items():
        expected = [[seq, chunk] for seq, n in enumerate(lengths)
                    for chunk in range((n + size - 1) // size)]
        assert indices.shape == (len(expected), 2)
        assert indices.tolist() == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_host_indices_do_not_wait_for_queued_gpu_work():
    prepare_kda_chunk_indices([6, 2304], device="cuda")
    torch.cuda.synchronize()
    with torch.device("cuda"):
        pending = torch.cuda.Event()
        torch.cuda._sleep(500_000_000)
        pending.record()
        result = prepare_kda_chunk_indices([2304, 5888], device="cuda")
        returned_early = not pending.query()
        torch.cuda.synchronize()
    assert returned_early, "Host KDA metadata synchronized queued GPU work"
    assert result[32].shape == (256, 2)
    assert result[64].shape == (128, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("lengths", [[6, 2304], [2304, 5888]])
@pytest.mark.parametrize("chunk_size", [None, 32, 64])
@pytest.mark.parametrize("safe_gate", [False, True])
def test_real_kda_matches_baseline_without_device_metadata_readback(
    monkeypatch, lengths, chunk_size, safe_gate
):
    from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn
    internal = importlib.import_module(
        "aiter.ops.triton._triton_kernels.chunk_delta_attn.chunk_fwd"
    )
    flash = importlib.import_module(
        "aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_kda"
    )
    torch.manual_seed(91)
    total, heads, dim = sum(lengths), 12, 128
    q, k, v, g = [torch.randn(1, total, heads, dim, device="cuda",
                              dtype=torch.bfloat16) * .1 for _ in range(4)]
    cu = torch.tensor([0, lengths[0], total], dtype=torch.int32, device="cuda")
    kwargs = dict(
        q=q, k=k, v=v, g=g,
        beta=torch.randn(1, total, heads, device="cuda", dtype=torch.float32),
        A_log=torch.zeros(heads, device="cuda", dtype=torch.float32),
        dt_bias=torch.zeros(heads * dim, device="cuda", dtype=torch.float32),
        initial_state=torch.randn(2, heads, dim, dim, device="cuda",
                                  dtype=torch.float32) * .01,
        output_final_state=True, use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
        safe_gate=safe_gate, lower_bound=-5.0 if safe_gate else None,
        cu_seqlens=cu, state_v_first=True, chunk_size=chunk_size,
    )
    baseline = chunk_kimi_delta_attn(**kwargs)
    indices = prepare_kda_chunk_indices(lengths, device="cuda", dtype=cu.dtype)

    def forbidden(*args, **kwargs):
        raise AssertionError("AITER tried to reconstruct indices from GPU lengths")

    monkeypatch.setattr(internal, "prepare_chunk_indices", forbidden)
    monkeypatch.setattr(flash, "_seq_bounds", forbidden)
    torch.cuda.synchronize()
    # Exercise uncached descriptor uploads, not a cache hit hiding a barrier.
    flash._build_segments.cache_clear()
    pending = torch.cuda.Event()
    torch.cuda._sleep(2_000_000_000)
    pending.record()
    actual = chunk_kimi_delta_attn(
        **kwargs, chunk_indices_by_size=indices,
        seq_bounds_cpu=prepare_kda_seq_bounds(lengths),
    )
    returned_early = not pending.query()
    torch.cuda.synchronize()
    assert returned_early, "Full KDA API synchronized prior GPU work"
    for expected, observed in zip(baseline, actual):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_k3_forwards_precomputed_metadata(monkeypatch):
    from atom.models.kimi_k3 import KimiKDAAttention
    import aiter.ops.triton.kimi_delta_attn as api
    seen = {}
    monkeypatch.setattr(api, "chunk_kimi_delta_attn", lambda **kw: seen.update(kw))
    module = SimpleNamespace(A_log=None, dt_bias=None, _kda_gate_lower_bound=None)
    tensor = torch.zeros(1)
    indices = prepare_kda_chunk_indices([6, 2304], device="cpu")
    KimiKDAAttention._run_kda(module, tensor, tensor, tensor, tensor, tensor,
                            None, tensor, True, indices, ((0, 6), (6, 2310)))
    assert seen["chunk_indices_by_size"] is indices
    assert seen["seq_bounds_cpu"] == ((0, 6), (6, 2310))
    assert "chunk_size" not in seen  # AITER still selects its original kernel.
