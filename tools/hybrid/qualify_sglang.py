"""GPU serving qualification against the pinned HF release at each midtrain size.

Each size runs in a separate process. JSON reports survive a later model failure.
This measures serving correctness and short throughput, not RL learning quality.
"""

import argparse
import gc
import hashlib
import json
import os
import signal
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

register()
os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "miles_plugins.hybrid.models"

STEPS = {"275m": "0038147", "450m": "0038147", "810m": "0023842", "1.4b": "0011921", "2.7b": "0011921"}
ROOT = "/weka/oe-training-default/yashasbls/hybridish-local/hf/v1"


def compare_reference(model, prompt, generated, meta):
    """Compare teacher-forced HF probabilities on the actual served decode path."""
    tokens = prompt + generated
    first = max(1, len(prompt) - 16)
    inputs = torch.tensor([tokens], device="cuda")
    with torch.inference_mode():
        logits = model(inputs, use_cache=False, logits_to_keep=len(tokens) - first + 1).logits[0].float()
    # The final logit predicts beyond the supplied sequence and has no target.
    scores = logits[:-1].log_softmax(-1)
    targets = torch.tensor(tokens[first:], device="cuda")
    expected = scores.gather(-1, targets[:, None]).squeeze(-1).cpu()
    observed = meta["input_token_logprobs"] + meta["output_token_logprobs"]
    observed = [entry for entry in observed if entry[0] is not None]
    if [entry[1] for entry in observed] != tokens[first:]:
        raise AssertionError("SGLang logprob alignment differs from the teacher-forced sequence")
    actual = torch.tensor([entry[0] for entry in observed])
    if not torch.isfinite(actual).all():
        raise AssertionError("Non-finite serving log probabilities")
    error = (expected - actual).abs().max().item()
    decode_scores = scores[-len(generated) :]
    chosen = decode_scores.gather(-1, targets[-len(generated) :, None]).squeeze(-1)
    gap = decode_scores.max(-1).values - chosen
    agreement = (gap <= 0.01).float().mean().item()
    if error > 0.08 or agreement < 0.95:
        raise AssertionError(f"HF parity failed: max logprob error={error}, greedy agreement={agreement}")
    return {"max_logprob_error": error, "greedy_agreement_within_0.01": agreement}


def generate_cases(engine, tokenizer, model):
    unit = tokenizer.encode(
        "Question: Natalia sold 48 clips in April and half as many in May. How many in total? Answer:",
        add_special_tokens=False,
    )
    lengths = [1, 3, 31, 127, 511, 2047, 3071]
    prompts = [(unit * ((length // len(unit)) + 1))[:length] for length in lengths]
    records = []
    # Mixed sequence lengths exercise packed prefill, decode scheduling, and slot reuse.
    for batch in (prompts[:3], prompts[3:6], prompts[6:], prompts[:3]):
        start = time.monotonic()
        outputs = engine.generate(
            input_ids=batch,
            sampling_params={"temperature": 0, "max_new_tokens": 32, "ignore_eos": True},
            return_logprob=True,
            logprob_start_len=[max(1, len(prompt) - 16) for prompt in batch],
        )
        elapsed = time.monotonic() - start
        for prompt, output in zip(batch, outputs, strict=True):
            meta = output["meta_info"]
            generated = [entry[1] for entry in meta["output_token_logprobs"]]
            if len(generated) != 32:
                raise AssertionError(f"Expected 32 cached decode tokens, got {len(generated)}")
            parity = compare_reference(model, prompt, generated, meta)
            records.append({"prompt_tokens": len(prompt), "decode_tokens": len(generated), **parity})
        records[-1]["batch_seconds"] = elapsed
        records[-1]["batch_decode_tokens_per_second"] = 32 * len(batch) / elapsed
    return records


def qualify(size, output):
    # Optional GPU dependency: import only in the model subprocess, after config registration.
    import sglang

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
        "status": "RUNNING",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    tokenizer = load_tokenizer(source)
    model = (
        AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    )
    started = time.monotonic()
    engine = None
    try:
        engine = sglang.Engine(
            model_path=source,
            dtype="bfloat16",
            tp_size=1,
            pp_size=1,
            attention_backend="triton",
            disable_cuda_graph=True,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            max_running_requests=8,
            context_length=4096,
            mem_fraction_static=0.45,
            random_seed=1,
        )
        result["startup_seconds"] = time.monotonic() - started
        result["cases"] = generate_cases(engine, tokenizer, model)
        updated = engine.update_weights_from_disk(source)
        if not updated[0]:
            raise AssertionError(f"Weight reload failed: {updated}")
        result["after_reload"] = generate_cases(engine, tokenizer, model)[:3]
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.size == "all":
        failures = []
        for size in STEPS:
            with (args.output_dir / f"{size}.log").open("w") as log:
                try:
                    run = subprocess.Popen(
                        [sys.executable, __file__, "--size", size, "--output-dir", str(args.output_dir)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    code = run.wait(timeout=1200)
                except subprocess.TimeoutExpired:
                    os.killpg(run.pid, signal.SIGKILL)
                    run.wait()
                    code = 124
            print(json.dumps({"size": size, "exit_code": code}), flush=True)
            if code:
                failures.append(size)
        (args.output_dir / "summary.json").write_text(
            json.dumps({"failed": failures, "sizes": list(STEPS)}, indent=2) + "\n"
        )
        return bool(failures)
    output = args.output_dir / f"{args.size}.json"
    try:
        qualify(args.size, output)
    except Exception:
        report = json.loads(output.read_text()) if output.exists() else {"size": args.size}
        report.update(status="FAILED", error=traceback.format_exc())
        output.write_text(json.dumps(report, indent=2) + "\n")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
