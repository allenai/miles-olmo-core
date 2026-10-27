"""Olmo KDA training with a resident, frozen SigLIP tower and full connector.

The existing Core MoE module owns only language parameters and optimizer state.
Visual features are added after fresh language embedding scale/normalization;
the patch-token embedding therefore still receives language-model gradients.
"""

import hashlib
import json
from pathlib import Path

import torch
from olmo_sglang.vision import config as vision_config
from olmo_sglang.vision import model as vision_model
from safetensors import torch as safetensors_torch
from torch import distributed as dist

from miles.backends.core_utils import moe_models


def model_config_from_hf(hf, options):
    vision_config.validate_vision_config(hf)
    return moe_models.model_config_from_hf(hf.text_config, options)


def prepare_model_config(config, hf, options):
    moe_models.prepare_model_config(config, hf.text_config, options)


def validate_training_options(args):
    core = args.olmo_core
    if not core.freeze_image_encoder or not core.freeze_connector:
        raise ValueError("The vision pilot requires both freeze_image_encoder and freeze_connector")
    if core.publication_mode != "barrier" or core.sequence_packing or core.model_config:
        raise ValueError("The vision pilot requires barrier publication, no packing and the checkpoint model config")
    if args.use_rollout_routing_replay or args.use_rollout_indexer_replay:
        raise ValueError("Vision routing/indexer replay is not qualified")
    if core.expert_publication != "per_expert":
        raise ValueError("The vision pilot requires per_expert publication")
    if getattr(args, "custom_generate_function_path", None) != "open_instruct.miles.datasets.vision_rollout.generate":
        raise ValueError("Vision training requires the image-aware custom_generate_function_path")
    if getattr(args, "apply_chat_template", False) or getattr(args, "multimodal_keys", None):
        raise ValueError("The image rollout plugin requires document prompts without generic multimodal preprocessing")


def read_hf_state(source):
    """Read local safetensors without instantiating a second multimodal model."""
    source = Path(source)
    index = source / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
        files = sorted(set(mapping.values()))
    else:
        mapping, files = None, ["model.safetensors"]
    state = {}
    for name in files:
        path = (source / name).resolve()
        if path.parent != source.resolve():
            raise ValueError("Checkpoint shards must live in the checkpoint directory")
        shard = safetensors_torch.load_file(str(path))
        if state.keys() & shard.keys():
            raise ValueError("Duplicate checkpoint tensors across shards")
        if mapping is not None and any(mapping.get(key) != name for key in shard):
            raise ValueError("Checkpoint shard differs from the safetensors index")
        state.update(shard)
    if mapping is not None and state.keys() != mapping.keys():
        raise ValueError("Checkpoint is missing indexed tensors")
    if any(not name.startswith(("language_model.", "vision_tower.", "connector.")) for name in state):
        raise ValueError("Unexpected multimodal checkpoint namespace")
    return state


def tensor_identity(state):
    """Hash names, geometry, dtype and exact bytes, including BF16 tensors."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_train_module(args, *, common, optim, hf_config, hf_state):
    visual = vision_model.OlmoVisionStack(hf_config, device=common["device"], dtype=torch.bfloat16)
    visual.load_weights((name, value) for name, value in hf_state.items() if not name.startswith("language_model."))
    visual.requires_grad_(False).eval()
    language = {name.removeprefix("language_model."): value for name, value in hf_state.items() if name.startswith("language_model.")}
    module = moe_models.build_train_module(args, common=common, optim=optim, hf_config=hf_config.text_config, hf_state=language)
    module._miles_vision = visual
    module._miles_vision_identity = tensor_identity(visual.state_dict())
    module._miles_image_patch_token_id = hf_config.image_patch_token_id
    return module


def conditioning_inputs(module, batch):
    """Recompute normalized text embeddings on every actor/reference forward."""
    items = batch.get("multimodal_train_inputs")
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
        raise ValueError("Vision training requires exactly one document's image tensors")
    item, tokens = items[0], batch["tokens"]
    if tokens.ndim != 2 or tokens.shape[0] != 1 or "doc_lens" in batch:
        raise ValueError("Packed or batched vision documents are not qualified")
    positions = item["image_positions"]
    if positions.dtype != torch.long or positions.ndim != 1 or not positions.numel():
        raise ValueError("Image positions must be a nonempty int64 vector")
    prompt_length = tokens.shape[1] - batch["response_lengths"][0]
    expected = (tokens[0, :prompt_length] == module._miles_image_patch_token_id).nonzero().flatten()
    if not torch.equal(positions, expected):
        raise ValueError("Image positions differ from the prompt's patch tokens")
    visual = module._miles_vision
    if visual.training or any(p.requires_grad for p in visual.parameters()):
        raise ValueError("The visual stack must remain frozen and in evaluation mode")
    parameter = next(visual.parameters())
    with torch.no_grad():
        features = visual.encode_images(
            item["images"].to(device=parameter.device, dtype=parameter.dtype),
            item["pooled_patches_idx"].to(parameter.device),
        )
    embeddings = module.model.forward_embed(tokens)
    if features.shape != (1, positions.numel(), embeddings.shape[-1]):
        raise ValueError("Visual features do not cover the prompt's image positions")
    # Out-of-place index_add keeps the LM embedding and norm gradient path intact.
    embeddings = embeddings.index_add(1, positions.to(embeddings.device), features.to(embeddings.dtype))
    return {"input_embeddings": embeddings}


def iter_export_state(module, hf, *, stream_moe=True, fused_experts=False, for_publication=False):
    for name, value in moe_models.iter_export_state(module, hf.text_config, stream_moe=stream_moe, fused_experts=fused_experts):
        yield "language_model." + name, value
    if not for_publication:
        yield from module._miles_vision.state_dict().items()


def save_native(module, path):
    metrics = moe_models.save_native(module, path)
    if dist.get_rank() == 0:
        state = {name: value.detach().cpu().contiguous() for name, value in module._miles_vision.state_dict().items()}
        if tensor_identity(state) != module._miles_vision_identity:
            raise ValueError("Frozen visual weights changed during training")
        safetensors_torch.save_file(state, str(Path(path) / "frozen-vision.safetensors"))
        (Path(path) / "frozen-vision.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sha256": module._miles_vision_identity,
                    "freeze_image_encoder": True,
                    "freeze_connector": True,
                }
            )
        )
    return metrics


def load_native(module, path, *, optim=True):
    manifest = json.loads((Path(path) / "frozen-vision.json").read_text())
    expected = {"schema_version": 1, "sha256": module._miles_vision_identity, "freeze_image_encoder": True, "freeze_connector": True}
    if manifest != expected:
        raise ValueError("Checkpoint frozen visual identity or freeze policy differs")
    state = safetensors_torch.load_file(str(Path(path) / "frozen-vision.safetensors"))
    if tensor_identity(state) != manifest["sha256"]:
        raise ValueError("Checkpoint frozen visual tensors changed")
    module._miles_vision.load_state_dict(state, strict=True)
    moe_models.load_native(module, path, optim=optim)


def replay_context(module, batch):
    raise ValueError("Vision routing replay is not qualified")
