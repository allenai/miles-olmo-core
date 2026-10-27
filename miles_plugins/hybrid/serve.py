"""Register the release config in the server and spawned scheduler processes."""

import os
import runpy

from transformers_plugin import register_config

register_config()
os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "miles_plugins.hybrid.models"

if __name__ == "__main__":
    runpy.run_module("sglang.launch_server", run_name="__main__")
