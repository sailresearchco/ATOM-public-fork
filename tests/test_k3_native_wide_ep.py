from types import SimpleNamespace

import pytest
import torch

from atom.models import kimi_k3
from atom.model_engine.arg_utils import EngineArgs


@pytest.mark.parametrize('dp', [2, 4])
@pytest.mark.parametrize('tokens', [1, 7, 8, 9])
def test_receive_arena_keeps_tp_padding(monkeypatch, dp, tokens):
    from atom.model_ops.fused_moe import modular_kernel as mk
    context = SimpleNamespace(running_tokens=tokens, running_tokens_are_unified=True, is_prefill=False)
    monkeypatch.setattr(mk, 'get_forward_context', lambda: SimpleNamespace(context=context))
    monkeypatch.setattr(mk, 'get_dp_group', lambda: SimpleNamespace(world_size=dp))
    per_rank = (tokens + 7) // 8
    bound = per_rank * 8 * dp
    arena = torch.zeros(256, 4)
    dispatch_ids = torch.zeros(256, 16, dtype=torch.int32)
    kernel = SimpleNamespace(prepare_finalize=SimpleNamespace(num_dispatchers=lambda:dp*8))
    actual = mk.FusedMoEModularKernel._maybe_trim_dispatch_output(
        kernel, arena, None, dispatch_ids, dispatch_ids.float(),
        torch.zeros(per_rank, 16, dtype=torch.int32), None)
    assert actual[0].shape[0] == bound


def test_mixed_prefill_arena_uses_global_maximum(monkeypatch):
    from atom.model_ops.fused_moe import modular_kernel as mk
    context = SimpleNamespace(running_tokens=1, running_tokens_are_unified=False, is_prefill=False)
    forward = SimpleNamespace(context=context, dp_metadata=SimpleNamespace(max_tokens_across_dp=101))
    monkeypatch.setattr(mk, 'get_forward_context', lambda: forward)
    monkeypatch.setattr(mk, 'get_dp_group', lambda: SimpleNamespace(world_size=2))
    kernel = SimpleNamespace(prepare_finalize=SimpleNamespace(num_dispatchers=lambda:16))
    arena = torch.zeros(4096, 4)
    ids = torch.zeros(4096, 16, dtype=torch.int32)
    actual = mk.FusedMoEModularKernel._maybe_trim_dispatch_output(
        kernel, arena, None, ids, ids.float(), torch.zeros(1,16,dtype=torch.int32), None)
    assert actual[0].shape[0] == 101 * 16


def test_situ_missing_aiter_fusion_uses_existing_activation(monkeypatch):
    from aiter.ops import activation
    monkeypatch.delattr(activation, 'situv2_and_mul_quant', raising=False)
    layer = kimi_k3.SituAndMul(beta=4.0, linear_beta=25.0, fused_quant=True)
    assert not layer.fused_quant
    x = torch.randn(8, 512, device='cuda', dtype=torch.bfloat16)
    actual, scale = layer(x)
    gate, up = x.float().chunk(2, dim=-1)
    expected = 4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * 25 * torch.tanh(up / 25)
    assert scale is None
    torch.testing.assert_close(actual.float(), expected, atol=0.03, rtol=0.01)


@pytest.mark.parametrize('dp', [2, 4])
def test_native_cli_preserves_tp8_and_global_dp(dp):
    kwargs = EngineArgs(
        tensor_parallel_size=8, data_parallel_size=dp,
        data_parallel_size_local=1, enable_expert_parallel=True,
        moe_ep_flatten_tp_across_dp=True, all2all_backend='high-throughput',
    )._get_engine_kwargs()
    assert kwargs['tensor_parallel_size'] == 8
    assert kwargs['parallel_config'].data_parallel_size == dp
    assert kwargs['parallel_config'].data_parallel_size_local == 1
    assert kwargs['moe_ep_flatten_tp_across_dp']
    assert not kwargs['enable_dp_attention']
    assert kwargs['moe_all2all_backend'] == 'mori'


@pytest.mark.parametrize('tokens', [1, 7, 8, 9, 64, 67])
def test_tp_token_partition_and_reassembly(monkeypatch, tokens):
    """Each real token is sent once and output order survives TP padding."""
    x = torch.arange(tokens * 4, dtype=torch.float32).reshape(tokens, 4)
    logits = torch.arange(tokens * 2, dtype=torch.float32).reshape(tokens, 2)
    reference = x * 2 + logits.sum(dim=1, keepdim=True)
    per_rank = (tokens + 7) // 8
    padded = torch.nn.functional.pad(reference, (0, 0, 0, per_rank * 8 - tokens))
    sent = []
    monkeypatch.setattr(kimi_k3, 'get_current_atom_config', lambda: SimpleNamespace(moe_ep_flatten_tp_across_dp=True))
    for rank in range(8):
        class Experts:
            moe_parallel_config = SimpleNamespace(use_all2all_kernels=True)

            def __call__(self, a, b):
                sent.append(a.clone())
                return a * 2 + b.sum(dim=1, keepdim=True)

        def gather(result, dim):
            assert dim == 0
            torch.testing.assert_close(result, padded[rank * per_rank:(rank + 1) * per_rank])
            return padded

        monkeypatch.setattr(kimi_k3, 'get_tensor_model_parallel_rank', lambda: rank)
        monkeypatch.setattr(kimi_k3, 'get_tp_group', lambda: SimpleNamespace(all_gather=gather))
        block = SimpleNamespace(experts=Experts(), tp_size=8)
        actual = kimi_k3.KimiSparseMoeBlock._routed_moe(block, x, logits)
        torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(torch.cat(sent)[:tokens], x)


@pytest.mark.parametrize('combined', [False, True])
def test_k3_latent_norm_receives_full_routed_sum(monkeypatch, combined):
    """TP partials need a sum; MoRI-combined values must not be summed again."""
    value = torch.tensor([[3.0, 4.0]])
    calls = []
    def reduce(x):
        calls.append(1)
        return x * 8
    monkeypatch.setattr(kimi_k3, 'tensor_model_parallel_all_reduce', reduce)
    block = SimpleNamespace(
        tp_size=8, use_latent_moe=True, gate=lambda x: x,
        routed_expert_down_proj=lambda x: x,
        _routed_moe=lambda x, logits: x,
        experts=SimpleNamespace(moe_parallel_config=SimpleNamespace(use_all2all_kernels=combined)),
        routed_expert_norm=lambda x: x,
        routed_expert_up_proj=lambda x: x,
    )
    result = kimi_k3.KimiSparseMoeBlock.routed_expert_forward(block, value)
    torch.testing.assert_close(result, value if combined else value * 8)
    assert len(calls) == (0 if combined else 1)
