import pytest

from tests.fast.launch_scripts import sh_harness
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, REPO_ROOT_PLACEHOLDER, run_launch_script

_BACKGROUNDING_SCRIPT = """#!/bin/bash
set -ex
python3 -m sglang.launch_server --port 13141 >/dev/null 2>&1 &
curl -sf http://127.0.0.1:13141/health_generate
ray job submit --address="http://127.0.0.1:8265" -- python3 train.py
"""

_BACKGROUNDING_INSIDE_A_SUBSTITUTION_SCRIPT = """#!/bin/bash
set -ex
start_server() {
    python3 -m sglang.launch_server --port "$1" >/dev/null 2>&1 &
    echo "/tmp/server-$1.log"
}
LOG=$(start_server 13141)
curl -sf http://127.0.0.1:13141/health_generate
ray job submit --address="http://127.0.0.1:8265" -- python3 train.py "$LOG"
"""

_SYNTHETIC_SCRIPT = """#!/bin/bash
set -ex
EXPECTED_GPUS=32
while true; do
    AVAILABLE_GPUS=$(python3 -c "import ray; print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null || echo 0)
    if [ "$AVAILABLE_GPUS" -ge "$EXPECTED_GPUS" ]; then
        break
    fi
    sleep 5
done
hf download some/model --local-dir /root/models/some-model
torchrun --nproc-per-node 8 CHECKOUT/tools/convert_hf_to_torch_dist.py
ray job submit --address="http://127.0.0.1:8265" -- python3 CHECKOUT/train.py
"""


class TestRunLaunchScriptOnABackgroundingScript:
    @pytest.fixture
    def script(self, tmp_path):
        script = tmp_path / "backgrounding.sh"
        script.write_text(_BACKGROUNDING_SCRIPT)
        return script

    def test_a_backgrounded_command_is_still_recorded(self, script, tmp_path):
        """bash exits without reaping `&` children, so reading the capture too early loses them."""
        run = run_launch_script(script, sandbox=tmp_path / "sandbox", timeout=30)

        assert run.returncode == 0
        assert run.invocations_of("python3")[0][1:3] == ["-m", "sglang.launch_server"]

    @pytest.mark.parametrize("attempt", range(20))
    def test_backgrounding_does_not_perturb_the_recorded_order(self, script, tmp_path, attempt):
        """Snapshots assert an exact sequence, so a `&` must not shuffle records run to run."""
        run = run_launch_script(script, sandbox=tmp_path / f"sandbox-{attempt}", timeout=30)

        assert [argv[0] for argv in run.invocations] == ["python3", "curl", "ray"]

    def test_records_are_ordered_by_the_fork_that_made_them_not_by_arrival(self, tmp_path):
        """bash forks in command order, so the pid orders records even when a `&` child appends late."""
        late_background = f"11{sh_harness._ARG_SEPARATOR}python3{sh_harness._RECORD_SEPARATOR}"
        foreground = f"12{sh_harness._ARG_SEPARATOR}curl{sh_harness._RECORD_SEPARATOR}"

        parsed = sh_harness._parse_capture(foreground + late_background, sandbox=tmp_path)

        assert parsed == [["python3"], ["curl"]]

    def test_a_command_backgrounded_inside_a_substitution_does_not_hang_the_run(self, tmp_path):
        """Orphaned by its subshell, it lingers as a zombie wherever PID 1 does not reap, and killpg still sees the group."""
        script = tmp_path / "substitution.sh"
        script.write_text(_BACKGROUNDING_INSIDE_A_SUBSTITUTION_SCRIPT)

        run = run_launch_script(script, sandbox=tmp_path / "sandbox", timeout=30)

        assert run.returncode == 0
        assert [argv[0] for argv in run.invocations] == ["python3", "curl", "ray"]


class TestRunLaunchScriptEnvironmentFreeze:
    @pytest.fixture
    def script(self, tmp_path):
        script = tmp_path / "synthetic.sh"
        script.write_text(_SYNTHETIC_SCRIPT.replace("CHECKOUT", str(REPO_ROOT)))
        return script

    def test_extra_env_may_not_shadow_a_frozen_variable(self, script, tmp_path):
        """A caller overriding MASTER_ADDR would unfreeze every snapshot that records it."""
        with pytest.raises(AssertionError, match="MASTER_ADDR"):
            run_launch_script(script, sandbox=tmp_path / "sandbox", extra_env={"MASTER_ADDR": "10.0.0.9"})

    def test_extra_env_may_not_shadow_the_capture_channel(self, script, tmp_path):
        """Redirecting the capture path would make every shim record vanish silently."""
        with pytest.raises(AssertionError, match="MILES_SH_HARNESS_CAPTURE"):
            run_launch_script(
                script, sandbox=tmp_path / "sandbox", extra_env={"MILES_SH_HARNESS_CAPTURE": "/dev/null"}
            )

    def test_extra_env_may_still_supply_a_variable_the_harness_does_not_own(self, script, tmp_path):
        """The freeze must not block the per-script inputs the snapshot suite has to pass in."""
        run = run_launch_script(script, sandbox=tmp_path / "sandbox", extra_env={"BASE_FOLDER": "/frozen/checkpoints"})

        assert run.returncode == 0


class TestRunLaunchScriptOnTheShimEdgeCases:
    @pytest.fixture
    def run(self, tmp_path):
        script = tmp_path / "synthetic.sh"
        script.write_text(_SYNTHETIC_SCRIPT.replace("CHECKOUT", str(REPO_ROOT)))
        return run_launch_script(script, sandbox=tmp_path / "sandbox", timeout=30)

    def test_a_gpu_wait_loop_leaves_on_its_first_poll(self, run):
        """The python shim must emit a real number; emitting its repr spins the loop until timeout."""
        assert run.returncode == 0
        assert run.invocations_of("sleep") == []

    def test_downloads_and_torchrun_are_intercepted(self, run):
        """Unshimmed, these would pull real weights and start a real training job."""
        assert run.invocations_of("hf")[0][1] == "download"
        assert run.invocations_of("torchrun")[0][1] == "--nproc-per-node"

    def test_a_repo_path_inside_argv_becomes_a_placeholder(self, run):
        """Recordings must not embed the checkout location of whoever ran the test."""
        torchrun_argv = run.invocations_of("torchrun")[0]

        assert torchrun_argv[-1] == f"{REPO_ROOT_PLACEHOLDER}/tools/convert_hf_to_torch_dist.py"
        assert str(REPO_ROOT) not in " ".join(torchrun_argv)
