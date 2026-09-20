# SPDX-License-Identifier: Apache-2.0
"""Adapter contract checks with the real MorI config and CPU tensors."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from atom.model_ops.fused_moe import modular_kernel as mk
from atom.model_ops.fused_moe import mori_v2_prepare_finalize as v2
from atom.model_ops.fused_moe.mori_v2_config import MoriV2TransportConfig


@pytest.mark.parametrize("world", [16, 32])
@pytest.mark.parametrize("tokens", [1, 7, 8, 9, 101])
def test_v2_keeps_native_tp_padding(monkeypatch, world, tokens):
    forward = SimpleNamespace(
        context=SimpleNamespace(
            running_tokens=tokens, running_tokens_are_unified=True, is_prefill=False
        )
    )
    monkeypatch.setattr(mk, "get_forward_context", lambda: forward)
    monkeypatch.setattr(
        mk, "get_dp_group", lambda: SimpleNamespace(world_size=world // 8)
    )
    kernel = v2.MoriV2ModularKernel.__new__(v2.MoriV2ModularKernel)
    kernel.prepare_finalize = SimpleNamespace(num_dispatchers=lambda: world)
    arena = torch.zeros(4096, 4)
    ids = torch.zeros(4096, 16, dtype=torch.int32)
    rows = (tokens + 7) // 8
    output = kernel._maybe_trim_dispatch_output(
        arena, None, ids, ids.float(), torch.zeros(rows, 16, dtype=torch.int32), None
    )
    assert output[0].shape[0] == rows * world


@pytest.mark.parametrize("counts", [[1, 101], [1, 7, 8192, 7936], [0, 1, 9, 17]])
def test_v2_mixed_prefill_covers_all_senders(monkeypatch, counts):
    world = 8 * len(counts)
    monkeypatch.setattr(
        mk,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=SimpleNamespace(
                running_tokens=1, running_tokens_are_unified=False, is_prefill=True
            ),
            dp_metadata=SimpleNamespace(max_tokens_across_dp=max(counts)),
        ),
    )
    monkeypatch.setattr(
        mk, "get_dp_group", lambda: SimpleNamespace(world_size=len(counts))
    )
    monkeypatch.setattr(mk, "get_tp_group", lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(
        mk,
        "get_current_atom_config",
        lambda: SimpleNamespace(moe_ep_flatten_tp_across_dp=True),
    )
    kernel = v2.MoriV2ModularKernel.__new__(v2.MoriV2ModularKernel)
    kernel.prepare_finalize = SimpleNamespace(num_dispatchers=lambda: world)
    arena = torch.empty(131072, 1)
    ids = torch.empty(131072, 16, dtype=torch.int32)
    sizes = []
    for n in counts:
        result = kernel._maybe_trim_dispatch_output(
            arena,
            None,
            ids,
            ids,
            torch.empty((n + 7) // 8, 16, dtype=torch.int32),
            None,
        )
        sizes.append(result[0].shape[0])
    assert set(sizes) == {((max(counts) + 7) // 8) * world}
    assert min(sizes) >= sum(((n + 7) // 8) * 8 for n in counts)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_real_config_preserves_dtype_and_qp_cache_identity(monkeypatch, dtype):
    v2._import_v2()
    v2.init_mori_v2_op.cache_clear()
    comm = Mock()
    monkeypatch.setattr(v2, "_init_cco_comm", Mock(return_value=comm))
    made = []

    def construct(cfg, _comm):
        made.append(cfg)
        return SimpleNamespace(cfg=cfg)

    monkeypatch.setattr(v2, "EpDispatchCombineOp", construct)
    for qps in [1, 2, 1]:
        cfg = MoriV2TransportConfig.from_topology(
            world_size=16,
            local_ep_size=8,
            internode=True,
            environ={"ATOM_MORI_V2_QPS_PER_PE": str(qps)},
        )
        op = v2.init_mori_v2_op(
            ep_rank=0,
            ep_size=16,
            ep_src_global_rank=0,
            hidden_dim=3584,
            max_num_inp_token_per_rank=128,
            num_local_experts=56,
            num_experts_per_token=16,
            data_type=dtype,
            transport=cfg,
        )
        assert op.cfg.data_type == dtype
        assert op.cfg.gpu_per_node == 8
        assert op.cfg.is_internode
        assert op.cfg.num_qp_per_pe == qps
    assert len(made) == 2
    v2.init_mori_v2_op.cache_clear()


@pytest.mark.parametrize(
    "maximum,expected", [(1, "v2_ll"), (4096, "v2_ll"), (4097, "v2"), (8192, "v2")]
)
def test_auto_family_uses_global_tp_sender_bound(monkeypatch, maximum, expected):
    monkeypatch.setattr(
        v2,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=SimpleNamespace(
                running_tokens=1, is_prefill=False, running_tokens_are_unified=False
            ),
            dp_metadata=SimpleNamespace(max_tokens_across_dp=maximum),
        ),
    )
    monkeypatch.setattr(
        v2,
        "get_current_atom_config",
        lambda: SimpleNamespace(moe_ep_flatten_tp_across_dp=True),
    )
    monkeypatch.setattr(v2, "get_tp_group", lambda: SimpleNamespace(world_size=8))
    transport = MoriV2TransportConfig.from_topology(
        world_size=32, local_ep_size=8, internode=True, environ={}
    )
    op = SimpleNamespace(cfg=SimpleNamespace(internode_kernel="auto"))
    adapter = v2.MoriV2PrepareAndFinalize(op, 8192, 32, transport=transport)
    adapter._select_internode_family()
    assert op.cfg.internode_kernel == expected
    # Another layer sharing this op retains the original auto policy even after
    # the first layer changed the launch choice on the mutable MorI config.
    second = v2.MoriV2PrepareAndFinalize(op, 8192, 32, transport=transport)
    second._select_internode_family()
    assert op.cfg.internode_kernel == expected
