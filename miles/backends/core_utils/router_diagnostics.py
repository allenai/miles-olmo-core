"""Opt-in, rank-local routing observations from the actual replay forward.

These diagnostics synchronize to CPU and are unsuitable for throughput trials.
Fresh preferences are conditional on replayed upstream hidden states.
"""

import torch


def _summary(scores, weights, ids, fresh, margin):
    count, experts = scores.shape
    if not count:
        return {"tokens": 0}
    normalized = weights / weights.sum(-1, keepdim=True).clamp_min(1e-30)
    counts = torch.bincount(ids.flatten(), minlength=experts)
    mass = torch.zeros(experts).scatter_add_(0, ids.flatten(), normalized.flatten())
    overlap = (ids.unsqueeze(-1) == fresh.unsqueeze(-2)).any(-1).float().mean(-1)
    return dict(
        tokens=count,
        expert_counts=counts.tolist(),
        expert_mixing_mass=mass.tolist(),
        unique_experts=int((counts > 0).sum()),
        load_cv=float(counts.float().std(unbiased=False) / counts.float().mean()),
        full_gate_entropy=float(-(scores * scores.clamp_min(1e-30).log()).sum(-1).mean()),
        mixing_entropy=float(-(normalized * normalized.clamp_min(1e-30).log()).sum(-1).mean()),
        mixing_top1_mass=float(normalized.max(-1).values.mean()),
        selected_gate_mass=float(scores.gather(-1, ids).sum(-1).mean()),
        fresh_replay_overlap=float(overlap.mean()),
        fresh_replay_set_agreement=float((overlap == 1).float().mean()),
        selection_margin_mean=None if margin is None else float(margin.mean()),
        selection_margin_min=None if margin is None else float(margin.min()),
    )


@torch.no_grad()
def observe(router, output, lengths):
    """Summarize softmax full-pool routing; exclude each document's filler tail."""
    if (
        str(router.gating_function) not in ("softmax", "topk_softmax")
        or router.score_bias is not None
        or router.n_group is not None
        or router.uniform_expert_assignment
        or router.random_expert_assignment
    ):
        raise ValueError("Detailed router diagnostics require unbiased softmax full-pool selection")
    weights, ids, _, aux = output
    if aux is None:
        raise ValueError("Router diagnostics require Core's forward scores and logits")
    scores, logits = (value.detach().float().reshape(-1, router.num_experts) for value in aux[:2])
    weights = weights.detach().float().reshape(-1, router.top_k).cpu()
    ids = ids.detach().reshape(-1, router.top_k).cpu()
    if sum(lengths) != len(ids) or any(length < 1 for length in lengths):
        raise ValueError("Router diagnostic document lengths do not cover the forward")
    # Core selects topk_softmax from logits, and softmax from probabilities.
    selection = logits if str(router.gating_function) == "topk_softmax" else scores
    fresh = (
        selection.topk(router.top_k, dim=-1).indices
        if str(router.gating_function) == "topk_softmax"
        else router.get_top_k(scores)[1]
    )
    ordered = selection.topk(min(router.top_k + 1, router.num_experts), dim=-1).values
    margin = ordered[:, router.top_k - 1] - ordered[:, router.top_k] if router.top_k < router.num_experts else None
    scores, fresh = scores.cpu(), fresh.cpu()
    margin = None if margin is None else margin.cpu()
    documents, included, start = [], [], 0
    for index, length in enumerate(lengths):
        stop = start + length - 1
        token_indices = torch.arange(start, stop)
        included.append(token_indices)
        row = _summary(
            scores[start:stop],
            weights[start:stop],
            ids[start:stop],
            fresh[start:stop],
            None if margin is None else margin[start:stop],
        )
        documents.append(dict(document=index, **row))
        start += length
    keep = torch.cat(included)
    summary = _summary(scores[keep], weights[keep], ids[keep], fresh[keep], None if margin is None else margin[keep])
    sample = keep[:16]
    return dict(
        **summary,
        documents=documents,
        synthetic_tail_tokens=len(lengths),
        gating_function=str(router.gating_function),
        routing_mode="full_pool" if getattr(getattr(router, "emo", None), "full_pool", False) else "replay",
        expert_count=router.num_experts,
        top_k=router.top_k,
        token_sample=dict(
            positions=sample.tolist(),
            expert_ids=ids[sample].tolist(),
            weights=weights[sample].tolist(),
            fresh_expert_ids=fresh[sample].tolist(),
        ),
    )
