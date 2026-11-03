import subprocess
import sys
import textwrap
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock


def test_standard_tracking_does_not_require_open_instruct():
    # A fresh interpreter also catches eager imports when Open Instruct happens
    # to be installed in the developer's environment.
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                from types import SimpleNamespace
                from unittest.mock import Mock

                sys.modules["open_instruct"] = None
                from miles.utils.tracking_utils import wandb_utils

                wandb_utils.wandb.define_metric = Mock()
                wandb_utils._init_wandb_common(SimpleNamespace())
                wandb_utils.wandb.define_metric.assert_any_call("train/*", step_metric="train/step")
                wandb_utils.wandb.define_metric.assert_any_call("eval/*", step_metric="eval/step")
                """
            ),
        ],
        check=True,
        timeout=30,
    )


def test_background_evaluation_defines_its_own_metrics(monkeypatch):
    from miles.utils.tracking_utils import wandb_utils

    open_instruct = ModuleType("open_instruct")
    open_instruct.miles = ModuleType("open_instruct.miles")
    open_instruct.miles.evaluation = ModuleType("open_instruct.miles.evaluation")
    evaluation_runner = SimpleNamespace(define_metrics=Mock())
    open_instruct.miles.evaluation.evaluation_runner = evaluation_runner
    monkeypatch.setitem(sys.modules, "open_instruct", open_instruct)
    monkeypatch.setitem(sys.modules, "open_instruct.miles", open_instruct.miles)
    monkeypatch.setitem(sys.modules, "open_instruct.miles.evaluation", open_instruct.miles.evaluation)
    run = object()
    monkeypatch.setattr(wandb_utils.wandb, "run", run)
    define_metric = Mock()
    monkeypatch.setattr(wandb_utils.wandb, "define_metric", define_metric)

    wandb_utils._init_wandb_common(SimpleNamespace(background_evaluation=True))

    evaluation_runner.define_metrics.assert_called_once_with(run)
    define_metric.assert_not_called()
