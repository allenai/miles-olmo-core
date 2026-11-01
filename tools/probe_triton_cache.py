"""Bounded fresh-process cache qualification on one GPU; no model downloads.

Run: python tools/probe_triton_cache.py --output /output/cache-probe
Exercises a changed kernel, KDA forward/backward, and FA4 forward/backward.
Reports compilation separately from process startup and checks numerical reuse.
"""

import argparse
import collections
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from miles.utils.compiler_cache import compiler_cache as cache
from miles.utils.compiler_cache import startup_cache

KERNEL = """import triton
from triton import language as tl

@triton.jit
def helper(x):
    return x * FACTOR

@triton.jit
def transform(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, other=0)
    tl.store(Y + offsets, helper(values), offsets < N)
"""


def _triton_workload(root, phase):
    # Backend imports are deliberately after setup_worker installs the caches.
    torch = importlib.import_module("torch")
    kda = importlib.import_module("fla.ops.kda")
    path = root / "probe_kernel.py"
    factor = 3 if phase == "kernel-change" else 2
    path.write_text(KERNEL.replace("FACTOR", str(factor)))
    spec = importlib.util.spec_from_file_location("probe_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    n = 2048 if phase == "shape-change" else 1024
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    module.transform[((n + 255) // 256,)](x, y, n, 256)
    torch.testing.assert_close(y, x * factor, rtol=0, atol=0)
    torch.manual_seed(17)
    saved = {}
    # Different packed sequence counts exercise variable-length trainer kernels.
    for count in (1, 4):
        print(f"PROBE_KDA_START {phase} sequences={count}", flush=True)
        shape = (1, 512, 4, 128)
        q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        g = torch.full(shape, -0.1, device="cuda", dtype=torch.float32, requires_grad=True)
        beta = torch.full(shape[:-1], 0.1, device="cuda", dtype=torch.float32, requires_grad=True)
        cu_cpu = torch.arange(0, 513, 512 // count, dtype=torch.int32)
        out, _ = kda.chunk_kda(
            q, k, v, g, beta, cu_seqlens=cu_cpu.cuda(), cu_seqlens_cpu=cu_cpu, use_qk_l2norm_in_kernel=True
        )
        out.float().square().mean().backward()
        for name, tensor in [("output", out), ("q_grad", q.grad), ("k_grad", k.grad), ("v_grad", v.grad)]:
            assert torch.isfinite(tensor).all(), name
            saved[f"{count}/{name}"] = tensor.detach().cpu()
    print(f"PROBE_SERVING_START {phase}", flush=True)
    saved.update(_serving_workload(torch))
    torch.cuda.synchronize()
    return saved


def _serving_workload(torch):
    moe = importlib.import_module("sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe")
    torch.manual_seed(23)
    x = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(4, 512, 128, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(4, 128, 256, device="cuda", dtype=torch.bfloat16) * 0.02
    ids = torch.arange(32, device="cuda", dtype=torch.int32).reshape(16, 2) % 4
    weights = torch.full((16, 2), 0.5, device="cuda")
    config = moe.MoeRunnerConfig(num_experts=4, num_local_experts=4, top_k=2, inplace=False)
    out = moe.fused_experts(x, w1, w2, SimpleNamespace(topk_weights=weights, topk_ids=ids), config)
    expected = torch.zeros_like(x, dtype=torch.float32)
    for row in range(16):
        for sibling in range(2):
            expert = int(ids[row, sibling])
            gate, up = (w1[expert].float() @ x[row].float()).chunk(2)
            expected[row] += 0.5 * (w2[expert].float() @ (torch.nn.functional.silu(gate) * up))
    torch.testing.assert_close(out.float(), expected, atol=0.002, rtol=0.03)
    return {"serving_moe": out.cpu()}


def _fa4_workload():
    torch = importlib.import_module("torch")
    attention = importlib.import_module("flash_attn.cute")
    torch.manual_seed(17)
    q, k, v = [
        torch.randn((1, 256, 4, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)
    ]
    out = attention.flash_attn_func(q, k, v, causal=True)
    out.float().square().mean().backward()
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.detach().transpose(1, 2), k.detach().transpose(1, 2), v.detach().transpose(1, 2), is_causal=True
    ).transpose(1, 2)
    torch.testing.assert_close(out, expected, atol=0.02, rtol=0.02)
    torch.cuda.synchronize()
    return {"output": out.detach().cpu(), "q_grad": q.grad.cpu(), "k_grad": k.grad.cpu(), "v_grad": v.grad.cpu()}


def _child(root, phase):
    print(f"PROBE_PHASE_START {phase}", flush=True)
    started = time.monotonic()
    reports = root / phase / "reports"
    reports.mkdir(parents=True)
    compiler_root = Path(importlib.util.find_spec("triton").submodule_search_locations[0])
    descriptor = dict(
        shared=str(root / "shared"),
        report_dir=str(reports),
        slot="trainer-engine-actor-0-0",
        image="sha256:" + "a" * 64,
        runtime_lock={"probe": phase},
        sources={name: {"sha256": phase} for name in ("miles", "open-instruct", "olmo-core", "olmo-sglang")},
        kernel_runtime={"triton": cache.source_identity(compiler_root), "native_sources": {}},
        model_config={"probe": "kda-4x128"},
        run_config={"core": {"prepared_batch_overlap": phase == "orchestration"}, "miles": {}},
        restore=True,
        observe=True,
        max_storage_bytes=2 * 1024**3,
    )
    os.environ[startup_cache.ENV] = json.dumps(descriptor)
    startup_cache.setup_worker()
    report = json.loads(next(reports.glob("*.json")).read_text())
    assert report["restore"]["status"] in {"miss", "hit"}, report
    setup_seconds = time.monotonic() - started
    print(f"PROBE_SETUP_DONE {phase} seconds={setup_seconds:.3f} restore={report['restore']}", flush=True)
    started = time.monotonic()
    values = _fa4_workload() if phase.startswith("fa4") else _triton_workload(root, phase)
    elapsed = time.monotonic() - started
    torch = importlib.import_module("torch")
    baseline = root / ("fa4-cold" if phase.startswith("fa4") else "cold") / "outputs.pt"
    if baseline.exists():
        previous = torch.load(baseline, weights_only=True)
        for name, value in values.items():
            torch.testing.assert_close(value, previous[name], rtol=0, atol=0)
    torch.save(values, root / phase / "outputs.pt")
    published = startup_cache.publish_worker(report, retain_local=True)
    events = []
    for path in Path(report["local"]).glob("activity-*.jsonl"):
        events.extend(json.loads(line) for line in path.read_text().splitlines())
    summary = dict(
        phase=phase,
        setup_seconds=setup_seconds,
        workload_seconds=elapsed,
        restore=report["restore"],
        fingerprint=report["fingerprint"],
        explanation=report.get("compatibility_explanation"),
        publish=published.get("publish"),
        events=dict(collections.Counter(e["event"] for e in events)),
        compile_seconds=sum(e.get("compiler_microseconds", 0) for e in events if e["event"] == "compile_miss") / 1e6,
        numerical_checks="passed",
    )
    (root / phase / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase")
    options = parser.parse_args()
    root = options.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if options.phase:
        _child(root, options.phase)
        return
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED="1",
        FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=str(root / "fa4-cache"),
    )
    results = []
    for phase in ("cold", "warm", "orchestration", "kernel-change", "shape-change", "fa4-cold", "fa4-warm"):
        command = [sys.executable, __file__, "--output", str(root), "--phase", phase]
        phase_env = dict(env)
        if phase.startswith("fa4"):
            private = root / phase / "private"
            (private / "fa4").mkdir(parents=True)
            phase_env["FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"] = str(private / "fa4")
            if phase == "fa4-warm":
                restored = cache.restore(root / "fa4-shared", private, cache.digest("fa4-probe"), "fa4")
                assert restored["status"] == "hit", restored
        subprocess.run(command, env=phase_env, check=True, timeout=600)
        if phase == "fa4-cold":
            published = cache.publish(root / "fa4-shared", private, cache.digest("fa4-probe"), "fa4")
            assert published["status"] == "published", published
            (root / "fa4-publication.json").write_text(json.dumps(published, indent=2) + "\n")
        results.append(json.loads((root / phase / "summary.json").read_text()))
    assert results[0]["events"].get("compile_miss", 0) > 0, results[0]
    for result in results[1:3]:
        assert result["restore"]["status"] == "hit", result
        assert result["events"].get("compile_miss", 0) == 0, result
    for result in results[3:5]:
        assert result["events"].get("compile_miss", 0) > 0, result
    (root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print("CACHE_PROBE_PASSED", flush=True)


if __name__ == "__main__":
    main()
