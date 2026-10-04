"""The backend's runtime implementation must not import the application."""

import subprocess
import sys

import pytest


def test_backend_imports_without_open_instruct():
    pytest.importorskip("olmo_core")
    script = """
import importlib.abc
import pkgutil
import sys

class RejectApplication(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'open_instruct' or fullname.startswith('open_instruct.'):
            raise AssertionError('backend imported application: ' + fullname)
sys.meta_path.insert(0, RejectApplication())
from miles.backends import core_utils
for module in pkgutil.walk_packages(core_utils.__path__, core_utils.__name__ + '.'):
    importlib.import_module(module.name)
from miles.ray.specs.train import _TRAINER_ACTOR_CLASSES
assert _TRAINER_ACTOR_CLASSES['olmo_core'] == 'miles.backends.core_utils.actor.OLMoCoreTrainRayActor'
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
