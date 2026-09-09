"""KV offset preparation must not wait on earlier cross-rank GPU work."""
import pytest
import torch

from atom.model_ops.attentions.aiter_mla import _initialize_prefill_kv_indptr


@pytest.mark.parametrize("bs", [0, 1, 3, 8])
def test_prefill_offsets_exclude_padding(bs):
    lengths = torch.arange(1, 11, dtype=torch.int32)
    offsets = torch.full((11,), -1, dtype=torch.int32)
    _initialize_prefill_kv_indptr(offsets, lengths, bs)
    expected = [sum(range(1, i + 1)) for i in range(bs + 1)]
    assert offsets[:bs + 1].tolist() == expected
    assert offsets[bs + 1:].tolist() == [-1] * (10 - bs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_prefill_offsets_do_not_wait_for_queued_gpu_work():
    lengths = torch.arange(1, 11, dtype=torch.int32, device="cuda")
    offsets = torch.full((11,), -1, dtype=torch.int32, device="cuda")
    # Warm kernels and allocator paths before introducing the pending work.
    _initialize_prefill_kv_indptr(offsets, lengths, 8)
    torch.cuda.synchronize()
    pending = torch.cuda.Event()
    torch.cuda._sleep(500_000_000)
    pending.record()
    assert not pending.query(), "The queued delay finished before the probe"
    _initialize_prefill_kv_indptr(offsets, lengths, 8)
    returned_before_gpu_finished = not pending.query()
    torch.cuda.synchronize()
    assert returned_before_gpu_finished, "Metadata preparation synchronized the GPU"
    assert offsets[:9].tolist() == [0, 1, 3, 6, 10, 15, 21, 28, 36]
    assert offsets[9:].tolist() == [-1, -1]

    # Negative control: the original scalar assignment exhibits the barrier
    # seen in the native stack trace (HIP memcpy_and_sync).
    torch.cuda._sleep(500_000_000)
    pending.record()
    assert not pending.query()
    offsets[0] = 0
    original_assignment_waited = pending.query()
    torch.cuda.synchronize()
    assert original_assignment_waited
