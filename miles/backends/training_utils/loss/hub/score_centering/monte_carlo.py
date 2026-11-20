"""IID behavior-policy probes for score centering; repeated draws retain their weight."""

import torch

from miles.backends.training_utils.loss.hub.score_centering.importance_sampling import importance_weights


def monte_carlo_loss(
    logp,
    logq,
    advantages,
    active,
    *,
    probe_logp=None,
    probe_logq=None,
    probe_valid=None,
    mode="none",
    tis_clip=2.0,
    mis_low=0.5,
    mis_high=5.0,
):
    """Return token losses and diagnostics, with fixed importance weights.

    Uses the existing MILES score-centering REINFORCE surrogate, replacing
    the truncated-head correction with an IID estimate of E_q[f(p/q) log p].
    Importance coefficients stay detached, as in the upstream estimator.
    """
    settings = dict(mode=mode, tis_clip=tis_clip, mis_low=mis_low, mis_high=mis_high)
    lp = torch.where(active, logp, 0.0)
    lq = torch.where(active, logq.detach(), 0.0)
    advantage = torch.where(active, advantages.detach(), 0.0)
    with torch.no_grad():
        weight = importance_weights(lp - lq, **settings)
    loss = -advantage * weight * lp
    metrics = {"sc_importance_weight": weight.detach()}
    if probe_logp is not None:
        if probe_logq is None or probe_valid is None or probe_logp.shape != probe_logq.shape:
            raise ValueError("SCMC requires aligned probe logp/logq and validity arrays")
        if probe_logp.shape[-1] not in (32, 64, 128) or probe_valid.shape != probe_logp.shape:
            raise ValueError("SCMC requires 32, 64 or 128 probes per position")
        if bool((active[:, None] & ~probe_valid).any()):
            raise ValueError("Every active SCMC position must contain every independent draw")
        plp = torch.where(active[:, None], probe_logp, 0.0)
        plq = torch.where(active[:, None], probe_logq.detach(), 0.0)
        with torch.no_grad():
            weights = importance_weights(plp - plq, **settings)
        correction = (weights * plp).mean(-1)
        loss = loss + advantage * correction
        metrics.update(sc_correction=correction.detach(), sc_probe_weight_mean=weights.mean(-1).detach())
    return loss, metrics
