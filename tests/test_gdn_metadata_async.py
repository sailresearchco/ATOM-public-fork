"""Convolution metadata must use scheduled host lengths without a GPU barrier."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.model_ops.attentions import gdn_attn


@pytest.mark.parametrize("lengths", [[], [0, 1, 7, 8, 9], [8192], [2304, 5888], [2304, 5632]])
def test_conv_metadata_matches_chunk_layout(lengths):
    cu = torch.tensor([0] + np.cumsum(lengths).tolist(), dtype=torch.int32)
    args, batches, offsets = gdn_attn.compute_causal_conv1d_metadata(
        cu, seqlens_cpu=np.asarray(lengths, dtype=np.int32)
    )
    expected_batches, expected_offsets = [], []
    for request, length in enumerate(lengths):
        chunks = (length + 7) // 8
        expected_batches.extend([request] * chunks)
        expected_offsets.extend(range(chunks))
    count = len(expected_batches)
    assert args[8]["tot"] == count
    assert batches[:count].tolist() == expected_batches
    assert offsets[:count].tolist() == expected_offsets
    assert torch.all(batches[count:] == -1)
    assert torch.all(offsets[count:] == -1)


def test_native_builder_uses_prefill_lengths_without_decode_or_padding(monkeypatch):
    seen = {}

    def capture(cu, *, seqlens_cpu):
        seen["cu"] = cu.tolist()
        seen["lengths"] = list(seqlens_cpu)
        return None, None, None

    monkeypatch.setattr(gdn_attn, "compute_causal_conv1d_metadata", capture)
    buffer = SimpleNamespace(copy_to_gpu=lambda n: torch.zeros(n, dtype=torch.int32))
    builder = SimpleNamespace(
        use_spec_decode=False, replayssm=False, device="cpu",
        prepare_state_indices=lambda *a, **kw: None,
        non_spec_state_indices_tensor=buffer, non_spec_state_indices_in_tensor=buffer,
    )
    batch = SimpleNamespace(
        total_seqs_num_decode=2, total_seqs_num_prefill=2,
        total_tokens_num_decode=2, total_tokens_num_prefill=8192,
        total_seqs_num=4, total_tokens_num=8194,
        num_scheduled_tokens=np.array([2304, 5888, 1, 1], dtype=np.int32),
    )
    md = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 2304, 8192, 8193, 8194, 8194]),
        num_cached_tokens=None,
    )
    gdn_attn.GDNStateMixin.prepare_gdn_metadata(
        builder, batch, md, is_prefill=True, prepare_block_tables=False
    )
    assert seen == {"cu": [0, 2304, 8192], "lengths": [2304, 5888]}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_conv_metadata_returns_while_prior_gpu_work_is_pending():
    lengths = np.array([2304, 5888], dtype=np.int32)
    cu = torch.tensor([0, 2304, 8192], dtype=torch.int32, device="cuda")
    with torch.device("cuda"):
        gdn_attn.compute_causal_conv1d_metadata(cu, seqlens_cpu=lengths)
        torch.cuda.synchronize()
        pending = torch.cuda.Event()
        torch.cuda._sleep(500_000_000)
        pending.record()
        assert not pending.query()
        args, batches, offsets = gdn_attn.compute_causal_conv1d_metadata(
            cu, seqlens_cpu=lengths
        )
        returned_before_gpu_finished = not pending.query()
        torch.cuda.synchronize()
    assert returned_before_gpu_finished, "Convolution metadata synchronized the GPU"
    assert args[8]["tot"] == 1024
    assert batches[:1024].tolist() == [0] * 288 + [1] * 736
    assert offsets[:1024].tolist() == list(range(288)) + list(range(736))
    assert torch.all(batches[1024:] == -1)
    assert torch.all(offsets[1024:] == -1)
