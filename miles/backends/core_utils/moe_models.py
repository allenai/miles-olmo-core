"""Specialized OLMoDDP MoE training and Core-owned HF weight interchange."""

import torch
from olmo_core import config as core_config
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn import attention
from olmo_core.nn.hf import config as hf_config_utils
from olmo_core.nn.moe.v2 import olmo3, replay
from olmo_core.optim import OLMoDDPOptimizerConfig
from olmo_core.train.train_module import transformer as train_transformer
from olmo_core.train.train_module.transformer import config as train_config

from miles.backends.core_utils import data, fla_compat
from miles.backends.core_utils.timing import startup_stage


def register_hf_classes():
    hf_config_utils._register_olmo3moe_auto_classes()


def model_config_from_hf(hf, options):
    return olmo3.build_olmo3_moe_config_from_hf_config(
        hf,
        dtype=core_config.DType.bfloat16,
        attention_backend=attention.AttentionBackendName(options.attention_backend),
        router_aux_loss_weight=options.router_aux_loss_weight,
        router_z_loss_weight=options.router_z_loss_weight,
    )


def prepare_model_config(config, hf, options):
    if "linear_attention" in (getattr(hf, "layer_types", None) or []):
        fla_compat.install_kda_triton_compat()
    config.recompute_each_block = options.activation_checkpointing
    blocks = list(config.block.values()) if isinstance(config.block, dict) else [config.block]
    for block in [*blocks, *(config.block_overrides or {}).values()]:
        router = getattr(block, "routed_experts_router", None)
        if router is not None:
            if options.router_aux_count_source == "router_selected" and not hasattr(router, "lb_loss_count_source"):
                raise ValueError(
                    "Router-selected counts require the updated Core runtime; rebuild from runtime.lock.json"
                )
            router.lb_loss_count_source = options.router_aux_count_source
        experts = getattr(block, "routed_experts", None)
        if experts is not None:
            experts.row_specialization = options.row_specialization
            # Policy scoring must use the same SiLU rounding as eager training.
            experts.match_eager_rounding = True


class HFInitializedMoETrainModule(train_transformer.OLMoDDPTrainModule):
    """Import weights after Core sharding but before optimizer master initialization."""

    _miles_checkpoint_options: dict[str, bool | int]

    def __init__(self, *, hf_config, hf_state, startup_args=None, **kwargs):
        self._startup_args = startup_args
        self._initial_hf_config = hf_config
        self._initial_hf_state = hf_state
        super().__init__(**kwargs)
        self._initial_hf_state = None

    def init_model_weights(self, model_parts, max_sequence_length, rank_microbatch_size):
        with startup_stage(self._startup_args, "native_parameter_init", device=torch.cuda):
            super().init_model_weights(model_parts, max_sequence_length, rank_microbatch_size)
        if len(model_parts) != 1:
            raise ValueError("HF import requires a non-pipeline model")
        assert self._initial_hf_state is not None
        with startup_stage(self._startup_args, "hf_to_native", device=torch.cuda):
            olmo3.load_olmo3_moe_hf_state(model_parts[0], self._initial_hf_config, self._initial_hf_state)
        model_parts[0].refresh_rowwise_fp8_cache()


def build_train_module(args, *, common, optim, hf_config, hf_state):
    # MoE clipping is optimizer-owned; the shared dense constructor still uses this keyword.
    common = {key: value for key, value in common.items() if key != "max_grad_norm"}
    module = HFInitializedMoETrainModule(
        hf_config=hf_config,
        hf_state=hf_state,
        startup_args=args,
        **common,
        optim=OLMoDDPOptimizerConfig(**optim, max_grad_norm=args.clip_grad, compile=args.olmo_core.compile_optimizer),
        dp_config=train_config.TransformerDataParallelConfig(
            name=DataParallelType.ddp, use_reduce_scatter=args.olmo_core.use_reduce_scatter
        ),
        ep_config=(
            train_config.TransformerExpertParallelConfig(degree=args.olmo_core.expert_parallel_size)
            if args.olmo_core.expert_parallel_size > 1
            else None
        ),
    )

    module._miles_checkpoint_options = args.olmo_core.checkpoint_save_options()
    return module


def iter_export_state(module, hf, *, stream_moe=True, fused_experts=False):
    if fused_experts and not stream_moe:
        raise ValueError("Fused expert publication requires the streaming MoE export")
    if stream_moe:
        yield from olmo3.iter_olmo3_moe_hf_state(module.model, hf, fused_experts=fused_experts)
    else:
        yield from olmo3.gather_olmo3_moe_hf_state(module.model, hf, cpu=True).items()


def save_native(module, path):
    return module.save_state_dict_direct(str(path), **getattr(module, "_miles_checkpoint_options", {}))


def load_native(module, path, *, optim=True):
    options = getattr(module, "_miles_checkpoint_options", {})
    load_options = {key: options[key] for key in ("constant_memory_planning", "profile") if options.get(key)}
    module.load_state_dict_direct(str(path), load_optim_state=optim, **load_options)


def validate_training_options(args):
    """MoE routing replay and expert parallelism are supported by this backend."""


def validate_emo_training(config, hf, args):
    """Keep every EMO policy/reference forward on the same causal full-pool policy."""
    blocks = list(config.block.values()) if isinstance(config.block, dict) else [config.block]
    routers = [
        block.routed_experts_router
        for block in [*blocks, *(config.block_overrides or {}).values()]
        if getattr(block, "routed_experts_router", None) is not None
    ]
    has_emo = any(getattr(router, "emo", None) is not None for router in routers)
    active_metadata = any(
        getattr(hf, field, None) is not None
        for field in (
            "emo_min_document_expert_pool",
            "emo_max_document_expert_pool",
            "emo_eval_document_expert_pool",
            "emo_eos_token_id",
        )
    )
    if not has_emo and not active_metadata and getattr(hf, "emo_routing_mode", None) is None:
        return
    if getattr(hf, "emo_routing_mode", None) != "full_pool":
        raise ValueError("EMO RL requires an HF checkpoint prepared with emo_routing_mode='full_pool'")
    if not routers or any(router.emo is None or not router.emo.full_pool for router in routers):
        raise ValueError("Every EMO routed layer must use full-pool execution, including custom model configs")
    if not getattr(args, "use_rollout_routing_replay", False):
        raise ValueError("EMO RL v1 requires use_rollout_routing_replay")
    if args.olmo_core.router_aux_loss_weight or args.olmo_core.router_z_loss_weight:
        raise ValueError("EMO RL v1 requires zero router_aux_loss_weight and router_z_loss_weight")
    if any(router.lb_loss_weight or router.z_loss_weight for router in routers):
        raise ValueError("EMO RL v1 does not support auxiliary coefficients in custom model configs")
    if getattr(hf, "emo_eval_document_expert_pool", None) != hf.n_routed_experts or any(
        router.num_experts != hf.n_routed_experts or router.top_k != hf.num_experts_per_tok for router in routers
    ):
        raise ValueError("EMO trainer and serving must agree on the full expert pool and top-k")
    if args.olmo_core.router_aux_count_source != "executed":
        raise ValueError("EMO RL v1 requires router_aux_count_source='executed'")
    if any(str(router.gating_function) != "softmax" or router.normalize_expert_weights != 1.0 for router in routers):
        raise ValueError("EMO serving requires softmax gating with L1-normalized expert weights")


def replay_context(module, batch):
    return replay.replay_routes(module.model, data.router_routes(module.model, batch))
