"""TP=1 correctness bridge: release FLA GDN state plus SGLang full-attention KV.

GDN states live outside SGLang's memory planner. Reserve memory for them with a
conservative mem_fraction_static and max_running_requests. Prefix sharing,
chunked prefill, graph capture, and speculative decoding are deliberately rejected.
"""

from collections.abc import Iterable

import torch
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel, get_server_args
from torch import nn
from transformers_plugin import modeling_mainline_ladder as release


def validate_runtime(parallel, args, quant_config):
    if parallel.tp_size != 1 or parallel.pp_size != 1 or quant_config is not None:
        raise ValueError("MainlineLadder bridge requires TP=1, PP=1 and unquantized BF16")
    if not args.disable_cuda_graph or not args.disable_radix_cache:
        raise ValueError("MainlineLadder bridge requires disable_cuda_graph and disable_radix_cache")
    if args.chunked_prefill_size != -1 or getattr(args, "speculative_algorithm", None):
        raise ValueError("MainlineLadder bridge requires chunked_prefill_size=-1 and no speculation")


class MainlineLadderForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        validate_runtime(get_parallel(), get_server_args(), quant_config)
        if torch.get_default_dtype() != torch.bfloat16:
            raise ValueError("MainlineLadder bridge requires BF16 model computation")
        self.config = config
        self.model = release.MainlineLadderModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.logits_processor = LogitsProcessor(config)
        self.attention = nn.ModuleDict(
            {
                str(i): RadixAttention(
                    config.num_attention_heads,
                    config.head_dim,
                    config.head_dim**-0.5,
                    num_kv_heads=config.num_key_value_heads,
                    layer_id=i,
                )
                for i, kind in enumerate(config.layer_types)
                if kind == "full_attention"
            }
        )
        self._request_states = {}

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def _states_for_batch(self, batch):
        slots = batch.req_pool_indices.tolist()
        if batch.forward_mode.is_decode():
            lengths = [1] * len(slots)
            missing = set(slots) - self._request_states.keys()
            if missing:
                raise RuntimeError(f"Decode without hybrid prefill state: {missing}")
        elif batch.forward_mode.is_extend():
            if any(batch.extend_prefix_lens_cpu):
                raise ValueError("Prefix reuse is unsupported by the hybrid bridge")
            lengths = batch.extend_seq_lens_cpu
            for slot in slots:
                self._request_states[slot] = release.MainlineLadderDynamicCache(self.config)
        else:
            raise ValueError(f"Unsupported hybrid forward mode: {batch.forward_mode}")
        return [(self._request_states[slot], length) for slot, length in zip(slots, lengths, strict=True)]

    def _linear_forward(self, layer, hidden, states):
        outputs = []
        offset = 0
        for state, length in states:
            outputs.append(layer.linear_attn(hidden[offset : offset + length].unsqueeze(0), cache_params=state)[0])
            offset += length
        if offset != hidden.shape[0]:
            raise ValueError("Hybrid request lengths do not match the packed tokens")
        return torch.cat(outputs, dim=0)

    def _attention_forward(self, layer, layer_id, hidden, batch):
        attn = layer.self_attn
        shape = (-1, self.config.num_attention_heads, self.config.head_dim)
        kv_shape = (-1, self.config.num_key_value_heads, self.config.head_dim)
        q = attn.q_norm(attn.q_proj(hidden).view(shape))
        k = attn.k_norm(attn.k_proj(hidden).view(kv_shape))
        v = attn.v_proj(hidden).view(kv_shape)
        output = self.attention[str(layer_id)](q, k, v, batch).reshape(hidden.shape[0], -1)
        if hasattr(attn, "attn_gate"):
            output = output * attn.attn_gate(hidden).float().sigmoid().to(output.dtype)
        return attn.o_proj(output)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        if input_ids.numel() == 0:
            raise ValueError("Empty/idle batches are unsupported by the TP=1 hybrid bridge")
        states = self._states_for_batch(forward_batch)
        hidden = self.model.embed_tokens(input_ids) if input_embeds is None else input_embeds
        hidden = self.model.embed_norm(hidden * self.config.embed_scale)
        for layer_id, layer in enumerate(self.model.layers):
            normalized = layer.input_layernorm(hidden)
            mixed = (
                self._linear_forward(layer, normalized, states)
                if layer.layer_type == "linear_attention"
                else self._attention_forward(layer, layer_id, normalized, forward_batch)
            )
            hidden = hidden + layer.post_attention_layernorm(mixed)
            hidden = hidden + layer.post_feedforward_layernorm(layer.mlp(layer.ffn_layernorm(hidden)))
        hidden = self.model.norm(hidden)
        return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        parameters = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            if name not in parameters:
                raise ValueError(f"Unexpected hybrid checkpoint tensor: {name}")
            default_weight_loader(parameters[name], weight)
            loaded.add(name)
        self._request_states.clear()
        return loaded


EntryClass = MainlineLadderForCausalLM
