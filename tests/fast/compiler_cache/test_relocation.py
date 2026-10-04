"""Import ordering and application identity must survive moving into MILES."""

import dataclasses
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.utils.compiler_cache import startup_cache as startup


@dataclasses.dataclass
class CacheConfig:
    compiler_cache_root: str
    compiler_cache: bool = True
    compiler_cache_restore: bool = True
    compiler_cache_diagnostics: bool = False
    compiler_cache_max_storage_bytes: int = 8 * 1024**3
    compiler_cache_publish_interval_seconds: float = 600


@pytest.mark.parametrize("packaged", [False, True])
def test_prepare_uses_explicit_application_root(tmp_path, monkeypatch, packaged):
    app = tmp_path / "application"
    lock_dir = app / ("build/runtime/miles" if packaged else "runtime/miles")
    lock_dir.mkdir(parents=True)
    lock = {"base_image": {"docker_id": "sha256:" + "a" * 64}}
    (lock_dir / "runtime.lock.json").write_text(json.dumps(lock))
    source = app / "open_instruct"
    source.mkdir()
    (source / "example.py").write_text("application = True\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "test"}')
    monkeypatch.setattr(
        startup.util, "find_spec", lambda name: SimpleNamespace(submodule_search_locations=[str(source)])
    )
    args = SimpleNamespace(olmo_core=CacheConfig(str(tmp_path / "shared")), hf_checkpoint=str(model))
    startup.prepare(args, application_root=app)
    policy = args.olmo_core_startup_cache
    assert policy["runtime_lock"] == lock
    assert policy["sources"]["open-instruct"] == startup.cache.source_identity(source)
    assert policy["model_config"] == {"model_type": "test"}
    assert policy["run_config"]["core"] == {}


def test_default_missing_mount_keeps_ordinary_cache(monkeypatch):
    monkeypatch.setattr(Path, "is_dir", lambda path: False)
    args = SimpleNamespace(olmo_core=CacheConfig(None))
    startup.prepare(args, application_root="/not-needed")
    assert args.olmo_core_startup_cache is None
    assert startup.DEFAULT_SHARED == "/weka/oe-training-default/open-instruct-compiler-cache/tmp-7d"
    startup.cache.validate_shared_root(Path(startup.DEFAULT_SHARED))


def test_hook_and_serving_entrypoint_import_without_application_or_training_stack():
    script = """
import importlib.abc
import runpy
import sys
from unittest import mock

class RejectHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'open_instruct', 'torch', 'transformers', 'sglang'}:
            raise AssertionError('premature dependency: ' + fullname)
sys.meta_path.insert(0, RejectHeavyImports())
from miles.utils.compiler_cache import startup_cache
hook = startup_cache.worker_runtime_env(type('Args', (), {'olmo_core_startup_cache': {'slot': 'train'}})(), 'train', {})['worker_process_setup_hook']
module, function = hook.rsplit('.', 1)
assert getattr(importlib.import_module(module), function) is startup_cache.setup_worker
real_run_module = runpy.run_module
events = []
with mock.patch.object(startup_cache, 'setup_worker', side_effect=lambda: events.append('setup')), mock.patch('importlib.import_module') as load, mock.patch.object(runpy, 'run_module') as run:
    load.return_value.prime_serving_modules.side_effect = lambda argv: events.append('prime')
    run.side_effect = lambda *a, **kw: events.append('serve')
    sys.argv = ['serving', '--model-path', '/model']
    real_run_module('miles.utils.compiler_cache.serving', run_name='__main__')
    load.assert_called_once_with('miles.utils.compiler_cache.hf_module_cache')
    load.return_value.prime_serving_modules.assert_called_once_with(['--model-path', '/model'])
    run.assert_called_once_with('sglang.launch_server', run_name='__main__')
assert events == ['setup', 'prime', 'serve']
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_diagnostic_child_imports_moved_hook(tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    policy = dict(
        shared=str(tmp_path / "shared"), report_dir=str(report_dir), slot="train", observe=True, restore=False
    )
    monkeypatch.setenv(startup.ENV, json.dumps(policy))
    # Identity failure is a supported cold fallback; diagnostics must still work.
    monkeypatch.setattr(startup.probes, "toolchain", lambda env: {})
    real_import = importlib.import_module
    fake_ray = SimpleNamespace(get_runtime_context=lambda: SimpleNamespace(get_node_id=lambda: "node"))
    monkeypatch.setattr(
        startup.importlib, "import_module", lambda name: fake_ray if name == "ray" else real_import(name)
    )
    monkeypatch.setattr(startup, "observe_triton", lambda root: None)
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    startup.setup_worker()
    report = json.loads(next(report_dir.glob("*.json")).read_text())
    script = "from triton.runtime.cache import FileCacheManager; assert FileCacheManager.put.__module__ == 'miles.utils.compiler_cache.startup_cache'"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    startup.publish_worker(report)
