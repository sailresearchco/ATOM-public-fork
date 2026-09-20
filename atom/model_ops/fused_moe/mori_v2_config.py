# SPDX-License-Identifier: Apache-2.0
"""Host-only topology and launch selection for MorI EPv2."""

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class MoriV2TransportConfig:
    gpu_per_node: int
    kernel_backend: str | None
    num_qp_per_pe: int = 2
    internode_kernel: str = "auto"
    internode_auto_ll_max_tokens: int = 512

    @classmethod
    def from_topology(
        cls,
        *,
        world_size: int,
        local_ep_size: int,
        internode: bool,
        fused: bool = False,
        environ: Mapping[str, str] | None = None,
    ):
        env = os.environ if environ is None else environ
        if world_size <= 0 or local_ep_size <= 0 or world_size % local_ep_size:
            raise ValueError("EPv2 requires a uniform, positive physical EP node size")
        if internode != (local_ep_size < world_size):
            raise ValueError(
                "EPv2 local_ep_size disagrees with the all2all physical-node probe: "
                f"{world_size=}, {local_ep_size=}, {internode=}"
            )
        if fused and internode:
            raise ValueError("ATOM_MORI_V2_FUSED supports intranode MegaMoE only")
        backend = env.get("MORI_V2_KERNEL_BACKEND") or None
        if backend not in (None, "hip", "flydsl"):
            raise ValueError(f"Unsupported MORI_V2_KERNEL_BACKEND={backend!r}")
        if internode:
            if env.get("ATOM_MORI_FP4_DISPATCH", "0") == "1":
                raise ValueError("Internode EPv2 does not support FP4 dispatch")
            if env.get("ATOM_MORI_COMBINE_QUANT", "none") != "none":
                raise ValueError("Internode EPv2 does not support quantized combine")
            if backend == "flydsl":
                raise ValueError("Internode EPv2 requires the HIP backend, not FlyDSL")
            backend = "hip"
        try:
            qps = int(env.get("ATOM_MORI_V2_QPS_PER_PE", "2"))
            threshold = int(env.get("ATOM_MORI_V2_LL_MAX_TOKENS", "512"))
        except ValueError as exc:
            raise ValueError("EPv2 QP count and LL threshold must be integers") from exc
        if qps < 1 or threshold < 0:
            raise ValueError("EPv2 requires QPs >= 1 and LL threshold >= 0")
        family = env.get("ATOM_MORI_V2_INTERNODE_KERNEL", "auto")
        if family not in ("auto", "v2", "v2_ll"):
            raise ValueError(f"Unsupported EPv2 internode family {family!r}")
        return cls(local_ep_size, backend, qps, family, threshold)

    def kwargs(self):
        return dict(vars(self))

    def family_for_batch(self, max_tokens: int | None, tp_size: int = 1) -> str:
        """Choose from a group-wide CPU token bound, never a rank-local count."""
        if self.internode_kernel != "auto":
            return self.internode_kernel
        if max_tokens is None:
            # Initialization without forward metadata must agree on every rank.
            return "v2"
        if max_tokens < 0 or tp_size < 1:
            raise ValueError("Invalid EPv2 batch token bound or TP size")
        sender_rows = (max_tokens + tp_size - 1) // tp_size
        return "v2_ll" if sender_rows <= self.internode_auto_ll_max_tokens else "v2"
