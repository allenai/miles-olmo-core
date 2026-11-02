"""Bounded one-GPU NB-specialization check: unchanged outputs/gradients and fewer kernels."""

import argparse
import collections
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from miles.backends.core_utils import fla_conv_compat
from miles.utils.compiler_cache import compilation_observer
from miles.utils.compiler_cache import compiler_cache as cache


def checksum(tensor, torch):
    assert torch.isfinite(tensor).all()
    value = tensor.detach().contiguous().cpu()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def child(root, phase):
    local = Path(os.environ["TRITON_CACHE_DIR"]).parent
    compilation_observer.install(local)
    torch = importlib.import_module("torch")
    triton = importlib.import_module("triton")
    conv = importlib.import_module("fla.modules.conv")
    kernels = importlib.import_module("fla.modules.conv.triton.kernels")
    if phase != "baseline":
        fla_conv_compat.install()
    # Use identical launch parameters for the numerical comparison. Production
    # retains the full tuner; the tiny trainer probe exercises that integration.
    for name in fla_conv_compat._KERNELS:
        getattr(kernels, name).fn.configs = [triton.Config({"BD": 64}, num_warps=4)]
    cases = [(length, 512, varlen, False) for varlen in (False, True) for length in (1024, 2048, 8192, 32768)]
    cases += [(257, 128, False, True), (257, 128, True, True), (8192, 8192, True, False), (32768, 8192, True, False)]
    records = []
    started = time.monotonic()
    for index, (length, width, varlen, extras) in enumerate(cases):
        torch.manual_seed(103 + index)
        x = (torch.randn(1, length, width, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_()
        weight = (torch.randn(width, 4, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_()
        boundaries = [0, length // 4, length // 2, length * 3 // 4, length] if varlen else [0, length]
        cpu = torch.tensor(boundaries, dtype=torch.int32) if varlen else None
        cu = cpu.cuda() if cpu is not None else None
        bias = torch.randn(width, device="cuda", dtype=torch.bfloat16, requires_grad=True) if extras else None
        residual = torch.randn_like(x, requires_grad=True) if extras else None
        initial = (
            torch.randn(len(boundaries) - 1, width, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            if extras
            else None
        )
        y, state = conv.causal_conv1d(
            x,
            weight,
            bias=bias,
            residual=residual,
            initial_state=initial,
            output_final_state=extras,
            activation=None if extras else "silu",
            backend="triton",
            cu_seqlens=cu,
            cu_seqlens_cpu=cpu,
        )
        loss = y.float().square().mean()
        if state is not None:
            loss = loss + state.float().square().mean()
        loss.backward()
        if not extras:
            with torch.no_grad():
                reference = torch.zeros_like(x, dtype=torch.float32)
                for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
                    for offset in range(4):
                        reference[:, begin + offset : end] += (
                            x[:, begin : end - offset].float() * weight[:, 3 - offset].float()
                        )
                reference = torch.nn.functional.silu(reference).to(y.dtype)
                torch.testing.assert_close(y, reference, atol=0.002, rtol=0.02)
        tensors = {"output": y, "dx": x.grad, "dw": weight.grad}
        if extras:
            tensors.update(state=state, db=bias.grad, dr=residual.grad, dh0=initial.grad)
        records.append(
            {"case": [length, width, varlen, extras], "checksums": {k: checksum(v, torch) for k, v in tensors.items()}}
        )
        print(f"CONV_CASE_PASSED {phase} {records[-1]['case']}", flush=True)
    events = [json.loads(line) for p in local.glob("activity-*.jsonl") for line in p.read_text().splitlines()]
    counts = collections.Counter(e["kernel"] for e in events if e["event"] == "compile_miss")
    summary = dict(phase=phase, seconds=time.monotonic() - started, records=records, compile_misses=dict(counts))
    (root / f"{phase}.json").write_text(json.dumps(summary, indent=2) + "\n")
    if phase == "dynamic":
        summary["publish"] = cache.publish(
            root / "shared", local, cache.digest("conv-nb-probe"), "triton", partial_triton=True
        )
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.phase:
        child(root, args.phase)
        return
    for phase in ("baseline", "dynamic", "warm"):
        local = Path(tempfile.mkdtemp(prefix="conv-nb-probe-"))
        (local / "triton").mkdir()
        if phase == "warm":
            assert cache.restore(root / "shared", local, cache.digest("conv-nb-probe"), "triton")["status"] == "hit"
        env = dict(os.environ, TRITON_CACHE_DIR=str(local / "triton"), PYTHONDONTWRITEBYTECODE="1")
        subprocess.run(
            [sys.executable, __file__, "--output", str(root), "--phase", phase], env=env, check=True, timeout=600
        )
    reports = [json.loads((root / f"{phase}.json").read_text()) for phase in ("baseline", "dynamic", "warm")]
    assert reports[0]["records"] == reports[1]["records"] == reports[2]["records"], "Convolution numerics changed"
    for name in fla_conv_compat._KERNELS:
        assert reports[1]["compile_misses"][name] < reports[0]["compile_misses"][name], reports
    assert not reports[2]["compile_misses"], reports[2]["compile_misses"]
    (root / "summary.json").write_text(
        json.dumps(
            {"numerics": "bitwise_equal", "phases": [{k: v for k, v in r.items() if k != "records"} for r in reports]},
            indent=2,
        )
        + "\n"
    )
    print("CONV_NB_PROBE_PASSED", flush=True)


if __name__ == "__main__":
    main()
