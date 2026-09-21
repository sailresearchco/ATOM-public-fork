"""Exercise the actual runner lifecycle methods without importing GPU backends."""
import ast
import os
from pathlib import Path
import time
import types
import unittest
from unittest.mock import Mock, patch


class SpyglassLifecycle(unittest.TestCase):
    def runner(self, calls):
        tree = ast.parse((Path(__file__).parents[1] / 'atom/model_engine/model_runner.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ModelRunner')
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('start_profiler', 'stop_profiler')]
        profiler = types.SimpleNamespace(__enter__=lambda: calls.append('native_start'), __exit__=lambda *a: calls.append('native_stop'))
        env = dict(os=os, time=time, logger=Mock(), envs=types.SimpleNamespace(ATOM_PROFILER_MORE=False),
                   torch_profiler=types.SimpleNamespace(profile=lambda **kw: profiler,
                       ProfilerActivity=types.SimpleNamespace(CPU=0, CUDA=1)))
        exec(compile(ast.Module(body=methods, type_ignores=[]), '<runner lifecycle>', 'exec'), env)
        runner = types.SimpleNamespace(profiler_dir='/tmp/probe-test', profiler=None,
                                       config=types.SimpleNamespace(model='fixture'), rank=0)
        return runner, env

    def test_enabled_and_disabled_lifecycle(self):
        for enabled in (False, True):
            calls = []
            module = types.ModuleType('spyglass.atom')
            module.start_probes = lambda: calls.append('probe_start')
            module.stop_probes = lambda: calls.append('probe_stop')
            package = types.ModuleType('spyglass')
            package.atom = module
            runner, env = self.runner(calls)
            with patch.dict(os.environ, {'SPYGLASS_PERF_HOOKS': 'hooks.json'} if enabled else {}, clear=True), patch.dict('sys.modules', {'spyglass': package, 'spyglass.atom': module}):
                env['start_profiler'](runner)
                env['start_profiler'](runner)  # Already started: no duplicate lifecycle.
                env['stop_profiler'](runner)
                env['stop_profiler'](runner)
            self.assertEqual(calls, ['native_start', 'probe_start', 'native_stop', 'probe_stop'] if enabled else ['native_start', 'native_stop'])

    def test_registration_precedes_construction_and_is_gated(self):
        tree = ast.parse((Path(__file__).parents[1] / 'atom/model_engine/model_runner.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ModelRunner')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        registration = next(n for n in init.body if isinstance(n, ast.If) and 'register_spyglass' in ast.unparse(n))
        construction = next(n for n in init.body if isinstance(n, ast.Expr) and '_build_and_load_model' in ast.unparse(n))
        self.assertLess(registration.lineno, construction.lineno)
        self.assertIn('SPYGLASS_CAPTURE', ast.unparse(registration.test))
        self.assertIn('SPYGLASS_PERF_HOOKS', ast.unparse(registration.test))
