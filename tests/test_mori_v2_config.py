# SPDX-License-Identifier: Apache-2.0
"""Host-only tests; loading by path avoids importing GPU libraries."""

import importlib.util
import sys
import unittest
from pathlib import Path

_PATH = Path(__file__).parents[1] / "atom/model_ops/fused_moe/mori_v2_config.py"
_SPEC = importlib.util.spec_from_file_location("mori_v2_config_under_test", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
Config = _MODULE.MoriV2TransportConfig


class TestMoriV2Config(unittest.TestCase):
    def test_physical_internode_selection(self):
        # Include 2x4: world_size <= 8 must never imply a single physical node.
        for world, local in [(8, 4), (16, 8), (32, 8)]:
            with self.subTest(world=world):
                cfg = Config.from_topology(
                    world_size=world,
                    local_ep_size=local,
                    internode=True,
                    environ={},
                )
                self.assertEqual(cfg.gpu_per_node, local)
                self.assertEqual(cfg.kernel_backend, "hip")
                self.assertEqual(cfg.num_qp_per_pe, 2)

    def test_local_backend_is_preserved(self):
        for backend in (None, "flydsl", "hip"):
            env = {} if backend is None else {"MORI_V2_KERNEL_BACKEND": backend}
            cfg = Config.from_topology(
                world_size=8,
                local_ep_size=8,
                internode=False,
                environ=env,
            )
            self.assertEqual(cfg.kernel_backend, backend)

    def test_rejects_false_topology(self):
        for world, local, remote in [
            (16, 8, False),
            (8, 8, True),
            (16, 3, True),
            (0, 0, False),
        ]:
            with (
                self.subTest(world=world, local=local, remote=remote),
                self.assertRaises(ValueError),
            ):
                Config.from_topology(
                    world_size=world,
                    local_ep_size=local,
                    internode=remote,
                    environ={},
                )

    def test_rejects_unsupported_internode_paths(self):
        for fused, env in [(True, {}), (False, {"MORI_V2_KERNEL_BACKEND": "flydsl"})]:
            with self.assertRaises(ValueError):
                Config.from_topology(
                    world_size=16,
                    local_ep_size=8,
                    internode=True,
                    fused=fused,
                    environ=env,
                )

    def test_knobs_are_in_the_handle_cache_key(self):
        configs = []
        for qp in (1, 2, 4, 8):
            configs.append(
                Config.from_topology(
                    world_size=32,
                    local_ep_size=8,
                    internode=True,
                    environ={
                        "ATOM_MORI_V2_QPS_PER_PE": str(qp),
                        "ATOM_MORI_V2_INTERNODE_KERNEL": "v2_ll",
                        "ATOM_MORI_V2_LL_MAX_TOKENS": "64",
                    },
                )
            )
        self.assertEqual(len(set(configs)), 4)
        self.assertEqual(
            configs[-1].kwargs(),
            {
                "gpu_per_node": 8,
                "kernel_backend": "hip",
                "num_qp_per_pe": 8,
                "internode_kernel": "v2_ll",
                "internode_auto_ll_max_tokens": 64,
            },
        )

    def test_bad_knobs_fail_before_collective_initialization(self):
        for key, value in [
            ("ATOM_MORI_V2_QPS_PER_PE", "0"),
            ("ATOM_MORI_V2_QPS_PER_PE", "1.5"),
            ("ATOM_MORI_V2_LL_MAX_TOKENS", "-1"),
            ("ATOM_MORI_V2_INTERNODE_KERNEL", "v1"),
            ("ATOM_MORI_FP4_DISPATCH", "1"),
            ("ATOM_MORI_COMBINE_QUANT", "fp8_blockwise"),
            ("MORI_V2_KERNEL_BACKEND", "unknown"),
        ]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                Config.from_topology(
                    world_size=16, local_ep_size=8, internode=True, environ={key: value}
                )


if __name__ == "__main__":
    unittest.main()
