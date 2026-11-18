"""Prepare a serialized rollout checkpoint using the Core publication quantizer.

Example: python tools/convert_core_hf_quantized.py --model-dir /models/bf16 \
    --save-dir /models/fp8 --format fp8 --device cuda

Input is Core's ordinary per-expert HF safetensors export. Training continues
from that original checkpoint; configure the new checkpoint for rollout only.
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import safetensors
import safetensors.torch
import torch

from miles.backends.core_utils.quantized_export import iter_quantized_state


def _inventory(source):
    index = source / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    path = source / "model.safetensors"
    with safetensors.safe_open(path, framework="pt", device="cpu") as reader:
        return dict.fromkeys(reader.keys(), path.name)


def _state(source, inventory, device):
    # Lexical order places each expert's down, gate, up consecutively, even when
    # its pair was stored in different input shards. Hold just one input tensor.
    for name in sorted(inventory):
        with safetensors.safe_open(source / inventory[name], framework="pt", device=device) as reader:
            yield name, reader.get_tensor(name)


def _write_weights(state, target, shard_bytes, *, nvfp4):
    bucket, weight_map, size, total, index = {}, {}, 0, 0, 0

    def flush():
        nonlocal index, size
        if not bucket:
            return
        filename = f"model-{index:05d}.safetensors"
        safetensors.torch.save_file(bucket, target / filename, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(bucket, filename))
        bucket.clear()
        size = 0
        index += 1

    for name, tensor in state:
        entries = [(name, tensor)]
        if nvfp4 and name.endswith(".weight_scale_2"):
            # Serialized NVFP4 initial loading also expects activation scales.
            # Match MILES' existing converter: dynamic activation uses unit scale.
            entries.append((name.replace(".weight_scale_2", ".input_scale"), torch.ones_like(tensor)))
        for key, value in entries:
            if size and size + value.nbytes > shard_bytes:
                flush()
            bucket[key] = value.detach().to(device="cpu", copy=True).contiguous()
            size += value.nbytes
            total += value.nbytes
    flush()
    if not weight_map:
        raise ValueError("Cannot export an empty checkpoint")
    (target / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n"
    )


def convert(source, destination, *, fmt, device="cuda", shard_bytes=256 * 1024**2):
    source, destination = Path(source), Path(destination)
    if fmt not in ("fp8", "nvfp4") or shard_bytes < 1:
        raise ValueError("Expected fp8/nvfp4 and positive shard_bytes")
    if destination.exists():
        raise FileExistsError(destination)
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "olmo3moe" or config.get("quantization_config"):
        raise ValueError("Input must be an unquantized Core Olmo MoE HF export")
    inventory = _inventory(source)
    ignored = sorted(
        name.removesuffix(".weight") for name in inventory if name.endswith(".weight") and ".mlp.experts." not in name
    )
    recipe = (
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
            "ignored_layers": ignored,
        }
        if fmt == "fp8"
        else {"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16, "exclude_modules": ignored}
    )
    config["quantization_config"] = recipe
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".quantized-export-", dir=destination.parent))
    try:
        _write_weights(
            iter_quantized_state(_state(source, inventory, device), recipe), staging, shard_bytes, nvfp4=fmt == "nvfp4"
        )
        for path in source.iterdir():
            if path.is_file() and (
                path.name.startswith("tokenizer")
                or path.suffix == ".py"
                or path.name
                in (
                    "generation_config.json",
                    "special_tokens_map.json",
                    "added_tokens.json",
                    "vocab.json",
                    "merges.txt",
                )
            ):
                shutil.copyfile(path, staging / path.name)
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        # Never expose a partially converted checkpoint as a loadable model.
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--format", choices=("fp8", "nvfp4"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-size-mb", type=int, default=256)
    args = parser.parse_args()
    convert(
        args.model_dir, args.save_dir, fmt=args.format, device=args.device, shard_bytes=args.shard_size_mb * 1024**2
    )


if __name__ == "__main__":
    main()
