"""Restore the serving compiler cache before importing SGLang or its models."""

import importlib
import runpy
import sys

from miles.utils.compiler_cache import startup_cache

if __name__ == "__main__":
    startup_cache.setup_worker()
    # Import Transformers only after configuring the worker's local caches.
    importlib.import_module("miles.utils.compiler_cache.hf_module_cache").prime_serving_modules(sys.argv[1:])
    runpy.run_module("sglang.launch_server", run_name="__main__")
