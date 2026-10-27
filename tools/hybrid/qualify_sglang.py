"""GPU serving qualification against the pinned HF release at each midtrain size.

Each size runs in a separate process. JSON reports survive a later model failure.
This measures serving correctness and short throughput, not RL learning quality.
"""

import argparse
import gc
import hashlib
from importlib.metadata import version
import json
import os
import signal
import shutil
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM
from transformers_plugin import modeling_mainline_ladder as release
from transformers_plugin import register

from miles.utils.processing_utils import load_tokenizer
from miles_plugins.hybrid.qualification import align_logprobs
from miles_plugins.hybrid.reference import FP32Head, cached_logprobs

register()
os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "miles_plugins.hybrid.models"

STEPS = {"275m": "0038147", "450m": "0038147", "810m": "0023842", "1.4b": "0011921", "2.7b": "0011921"}
ROOT = "/weka/oe-training-default/yashasbls/hybridish-local/hf/v1"


def compare_reference(model, prompt, generated, meta):
    """Compare teacher-forced HF probabilities on the actual served decode path."""
    tokens = prompt + generated
    positions, values = align_logprobs(prompt, generated, meta)
    first = min(positions)
    inputs = torch.tensor([tokens], device="cuda")
    with torch.inference_mode():
        logits = model(inputs, use_cache=False, logits_to_keep=len(tokens) - first + 1).logits[0].float()
    # The final logit predicts beyond the supplied sequence and has no target.
    full_scores = logits[:-1].log_softmax(-1)
    scores = cached_logprobs(model, prompt, generated, first_position=first)
    targets = torch.tensor([tokens[position] for position in positions], device="cuda")
    row_indices = torch.tensor([position - first for position in positions], device="cuda")
    expected = scores[row_indices].gather(-1, targets[:, None]).squeeze(-1).cpu()
    actual = torch.tensor(values)
    if not torch.isfinite(actual).all():
        raise AssertionError("Non-finite serving log probabilities")
    error = (expected - actual).abs().max().item()
    decode_scores = scores[-len(generated) :]
    chosen = decode_scores.gather(-1, targets[-len(generated) :, None]).squeeze(-1)
    gap = decode_scores.max(-1).values - chosen
    agreement = (gap <= 0.01).float().mean().item()
    full_expected = full_scores[row_indices].gather(-1, targets[:, None]).squeeze(-1).cpu()
    return {
        "passed": error <= 0.08 and agreement >= 0.95,
        "max_logprob_error": error,
        "greedy_agreement_within_0.01": agreement,
        "full_sequence_max_logprob_error": (full_expected - actual).abs().max().item(),
    }


def generate_cases(engine, tokenizer, model, records):
    unit = tokenizer.encode(
        "Question: Natalia sold 48 clips in April and half as many in May. How many in total? Answer:",
        add_special_tokens=False,
    )
    lengths = [1, 3, 31, 127, 511, 2047, 3071]
    prompts = [(unit * ((length // len(unit)) + 1))[:length] for length in lengths]
    # Mixed sequence lengths exercise packed prefill, decode scheduling, and slot reuse.
    batches = [(batch, [32] * len(batch)) for batch in (prompts[:3], prompts[3:6], prompts[6:], prompts[:3])]
    # Six requests exceed max_running_requests=4. Different completion lengths
    # force new prefills while other requests still own recurrent state.
    batches.append((prompts[:6], [8, 32, 16, 48, 8, 32]))
    for batch, decode_lengths in batches:
        start = time.monotonic()
        outputs = engine.generate(
            input_ids=batch,
            sampling_params=[{"temperature": 0, "max_new_tokens": count, "ignore_eos": True} for count in decode_lengths],
            return_logprob=True,
            logprob_start_len=[max(1, len(prompt) - 16) for prompt in batch],
        )
        elapsed = time.monotonic() - start
        for prompt, output, expected_length in zip(batch, outputs, decode_lengths, strict=True):
            meta = output["meta_info"]
            generated = [entry[1] for entry in meta["output_token_logprobs"]]
            if len(generated) != expected_length:
                raise AssertionError(f"Expected {expected_length} cached decode tokens, got {len(generated)}")
            parity = compare_reference(model, prompt, generated, meta)
            records.append({"prompt_tokens": len(prompt), "decode_tokens": len(generated), **parity})
        records[-1]["batch_seconds"] = elapsed
        records[-1]["batch_output_tokens_per_second"] = sum(decode_lengths) / elapsed
        records[-1]["queued_requests"] = len(batch) > 4
    return records


def probe_distributions(engine, tokenizer, model, records):
    """Compare every vocabulary probability against HF cached decode and full evaluation."""
    vocab = list(range(model.config.vocab_size))
    unit = tokenizer.encode(
        "Question: Natalia sold 48 clips in April and half as many in May. How many in total? Answer:",
        add_special_tokens=False,
    )
    for length in (1, 3, 511, 3071):
        decode_steps = 32 if length <= 3 else 2
        prompt = (unit * (length // len(unit) + 1))[:length]
        output = engine.generate(
            input_ids=prompt,
            sampling_params={"temperature": 0, "max_new_tokens": decode_steps, "ignore_eos": True},
            return_logprob=True,
            logprob_start_len=len(prompt) - 1,
            token_ids_logprob=vocab,
        )
        metadata = output["meta_info"]
        generated = [entry[1] for entry in metadata["output_token_logprobs"]]
        dense = metadata["output_token_ids_logprobs"]
        if len(dense) != decode_steps or any([entry[1] for entry in row] != vocab for row in dense):
            raise AssertionError("Full-vocabulary logprobs must cover every token at both steps")
        served = torch.tensor([[entry[0] for entry in row] for row in dense], device="cuda")
        with torch.inference_mode():
            inputs = torch.tensor([prompt + generated[:-1]], device="cuda")
            reference = model(inputs, use_cache=False, logits_to_keep=decode_steps).logits[0].float().log_softmax(-1)
        if not torch.isfinite(served).all() or served.logsumexp(-1).abs().max().item() > 1e-4:
            raise AssertionError("Serving full-vocabulary probabilities are not finite and normalized")
        cached = cached_logprobs(model, prompt, generated)
        if not torch.isfinite(reference).all() or not torch.isfinite(cached).all():
            raise AssertionError("Non-finite HF reference probabilities")
        full_kl = (served.exp() * (served - reference)).sum(-1).clamp_min(0)
        self_kl = (cached.exp() * (cached - reference)).sum(-1).clamp_min(0)
        kl = (served.exp() * (served - cached)).sum(-1).clamp_min(0)
        tv = (served.exp() - cached.exp()).abs().sum(-1) / 2
        record = {
            "passed": kl.max().item() <= 0.001,
            "prompt_tokens": length,
            "prefill_kl": kl[0].item(),
            "cached_decode_kl": kl[1:].max().item(),
            "decode_steps": decode_steps,
            "max_tv": tv.max().item(),
            "full_sequence_max_kl": full_kl.max().item(),
            "hf_cached_vs_full_max_kl": self_kl.max().item(),
        }
        records.append(record)


def qualify(size, output, distribution_only=False, fp32_logits=False, reference_backbone=False):
    # Optional GPU dependency: import only in the model subprocess, after config registration.
    import sglang

    os.environ["MILES_HYBRID_REFERENCE_BACKBONE"] = "1" if reference_backbone else "0"
    torch.set_num_threads(4)
    source = f"{ROOT}/{size}/midtrain/step{STEPS[size]}"
    config_bytes = (Path(source) / "config.json").read_bytes()
    descriptor = json.loads(config_bytes)
    assert descriptor["model_type"] == "mainline_ladder"
    assert descriptor["linear_allow_neg_eigval"] is True
    assert release.is_fast_path_available, "Release FLA fast path is required; no torch fallback qualification"
    result = {
        "size": size,
        "source": source,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256((Path(__file__).resolve().parents[2] / "miles_plugins/hybrid/models/mainline_ladder.py").read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "sglang": version("sglang"),
        "throughput_includes_prefill": True,
        "qualification_mode": "distribution_only" if distribution_only else "complete",
        "fp32_logits": fp32_logits,
        "serving_backbone": "hf_sdpa" if reference_backbone else "sglang_radix",
        "reference_mode": "hf_cached_decode_with_full_sequence_diagnostic",
        "status": "RUNNING",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    tokenizer = load_tokenizer(source)
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    if fp32_logits:
        model.lm_head = FP32Head(model.lm_head)
    started = time.monotonic()
    engine = None
    try:
        engine = sglang.Engine(
            model_path=source,
            dtype="bfloat16",
            tp_size=1,
            pp_size=1,
            attention_backend="triton",
            enable_fp32_lm_head=fp32_logits,
            disable_cuda_graph=True,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            max_running_requests=4,
            context_length=4096,
            mem_fraction_static=0.35,
            random_seed=1,
        )
        result["startup_seconds"] = time.monotonic() - started
        result["distribution_probes"] = []
        probe_distributions(engine, tokenizer, model, result["distribution_probes"])
        result["cases"] = []
        if not distribution_only:
            generate_cases(engine, tokenizer, model, result["cases"])
        updated = engine.update_weights_from_disk(source)
        if not updated[0]:
            raise AssertionError(f"Weight reload failed: {updated}")
        result["after_reload"] = []
        if not distribution_only:
            generate_cases(engine, tokenizer, model, result["after_reload"])
        result["distribution_probes_after_reload"] = []
        probe_distributions(engine, tokenizer, model, result["distribution_probes_after_reload"])
        checks = [record for key in ("distribution_probes", "cases", "after_reload", "distribution_probes_after_reload") for record in result[key]]
        if not all(record["passed"] for record in checks):
            raise AssertionError(f"Numerical parity failed in {sum(not record['passed'] for record in checks)} cases")
        result["status"] = "PASSED"
    finally:
        if engine is not None:
            engine.shutdown()
        result["total_seconds"] = time.monotonic() - started
        output.write_text(json.dumps(result, indent=2) + "\n")
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["all", *STEPS], default="all")
    parser.add_argument("--sizes", choices=list(STEPS), nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path)
    parser.add_argument("--distribution-only", action="store_true")
    parser.add_argument("--fp32-logits", action="store_true")
    parser.add_argument("--reference-backbone", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.size == "all":
        failures = []
        selected = args.sizes or list(STEPS)
        for size in selected:
            with (args.output_dir / f"{size}.log").open("w") as log:
                try:
                    run = subprocess.Popen(
                        [
                            sys.executable,
                            __file__,
                            "--size",
                            size,
                            "--output-dir",
                            str(args.output_dir),
                            *(["--distribution-only"] if args.distribution_only else []),
                            *(["--fp32-logits"] if args.fp32_logits else []),
                            *(["--reference-backbone"] if args.reference_backbone else []),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    code = run.wait(timeout=1200)
                except subprocess.TimeoutExpired:
                    os.killpg(run.pid, signal.SIGKILL)
                    run.wait()
                    code = 124
            if args.mirror_dir:
                shutil.copytree(args.output_dir, args.mirror_dir, dirs_exist_ok=True)
            print(json.dumps({"size": size, "exit_code": code}), flush=True)
            if code:
                failures.append(size)
        (args.output_dir / "summary.json").write_text(json.dumps({"failed": failures, "sizes": selected}, indent=2) + "\n")
        if args.mirror_dir:
            shutil.copytree(args.output_dir, args.mirror_dir, dirs_exist_ok=True)
        return bool(failures)
    output = args.output_dir / f"{args.size}.json"
    try:
        qualify(args.size, output, distribution_only=args.distribution_only, fp32_logits=args.fp32_logits, reference_backbone=args.reference_backbone)
    except Exception:
        report = json.loads(output.read_text()) if output.exists() else {"size": args.size}
        report.update(status="FAILED", error=traceback.format_exc())
        output.write_text(json.dumps(report, indent=2) + "\n")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
