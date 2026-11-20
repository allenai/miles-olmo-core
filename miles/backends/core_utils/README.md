# OLMo-core backend

`actor.OLMoCoreTrainRayActor` implements the MILES training actor interface using
OLMo-core. MILES selects it directly for `train_backend=olmo_core`, retrieves the
rollout references through its ordinary trainer controller, and computes the RL
objective with its shared loss implementation.

The backend owns model construction, sample/batch alignment, optimizer updates,
native checkpoints, HF export, expert packing, router diagnostics, and the
`publication/` and `rollout/` packages. Publication retains the existing barrier,
mixed-policy refresh, and independent engine-drain modes. This relocation does
not expand supported architectures or topologies.

Open Instruct supplies the run configuration and driver, dataset preparation,
reward callbacks, data-source selection, rollout logging, Beaker launch, and
external evaluation. The native Core argument loader is still the explicit Open
Instruct configuration integration. Backend runtime modules themselves do not
import Open Instruct; `test_adapter_imports.py` enforces this in a Core runtime.

Two application callbacks preserve existing behavior:

- `args.core_records_factory`: import path to a factory accepting `args` and
  returning a recorder with `record_group` and `record_disposition`. Required
  when `args.olmo_core.records_root` is set; unused when recording is disabled.
  `miles.utils.inference_records.Recorder(args, lineage_factory=...)` owns the
  writer and record format. The application factory supplies checkpoint identity;
  its sampling/verifier metadata retains the existing format.
  `python -m miles.utils.record_summary summarize STORE --output DIR` reads the
  store using only standard-library dependencies. Prompt filtering and selection
  remain application policy.
- `args.core_evaluation_snapshot`: optional import path to a callable accepting
  `(args, completed_steps)` after each optimizer update and returning an HF export
  path or `None`. The backend synchronizes and exports the snapshot on all ranks;
  application code owns scheduling/submitting external evaluations.

The configuration object must supply the existing Core fields and
`checkpoint_save_options()`; the standard trainer compares writer options with
its configuration type's defaults. Source compatibility is preserved for the
Open Instruct configuration. GPU execution still requires the qualified Core and
serving revisions from that application's runtime lock.

Small planning/artifact algorithms also exist in the application's CPU-only
configuration and analysis tools, which cannot depend on this private runtime.
Cross-repository tests compare their ASTs and results so these boundaries cannot
drift silently. Shared configuration packaging can be reconsidered separately.

Tests of pure/backend utilities live in `tests/fast/backends/core_utils/`.
Application configuration, model/runtime integration, and numerical/distributed
checks remain in Open Instruct's dedicated `tests/miles/` image suite. That suite
also checks callbacks, mirrored CPU contracts and native hooks. GPU mechanics
runs must use a newly built image containing both matching source revisions.

The move intentionally preserves existing large training methods and lifecycle
state. It changes imports, module paths, application callback boundaries, HF
loader calls to MILES helpers, and model-validation exceptions to `ValueError`;
it does not redesign training or publication algorithms.

## When an answer hits the length limit

Suppose the model generates two answers to one prompt. One finishes and earns a
reward of 1. The other reaches the length limit halfway through. Open Instruct
offers two ways to treat the unfinished answer when training with GRPO.

| Choice | Effect on training if the group is retained |
|---|---|
| Give the unfinished answer zero reward (`core.reward_zero_truncated=true`) | Both answers enter the group comparison. Their mean reward is 0.5, so the finished answer scores above the mean and the unfinished one below it. The generated tokens remain eligible for the policy loss. |
| Mask the unfinished answer (`exclude_truncated`, with `core.reward_zero_truncated=false`) | The unfinished answer contributes no tokens to the policy loss and its reward is left out of the group comparison. In this example the finished answer is alone, so its reward equals the group mean. |

**Core decides whether to take an optimizer step from the response masks across
all training ranks.** If every response token in that update is masked, Core
consumes the batch, records an `optimizer_skipped` event, and skips the entire
update. Optimizer momentum, the learning-rate schedule, and the policy version
stay where they were. Zero reward alone never triggers this skip. A rank with
only masked tokens still participates if another rank has included tokens.

**Combining masking with auxiliary losses.** Router load balancing and router
z-loss use model tokens, including prompt tokens. Forced-exit guidance can
train on a stopping probe with nonzero advantage and parent weight. In an
update that contains some included response tokens, these objectives can still
use data from a masked response. In a fully masked update, Core skips them
along with the policy loss, even if they could produce a gradient on their own.
If you need router or guidance updates on all-truncated batches, keep the
responses eligible for training and select the zero-reward policy. Group filters
still apply before training and can discard an all-zero-reward group.

## Score centering

The Core actor dispatches `loss_type=score_centering` through the shared native
loss. Record generation-time candidates with `rollout_top_logprobs_num > 0`, use
`advantage_estimator=grpo`, and select `score_centering_is=tis` for truncated
importance weighting instead of `use_tis`. Native validation enables the FP32
serving head, following upstream [#4031](https://github.com/radixark/miles/pull/4031).

Complete sampling-support replay explicitly uses an empty-tail ratio of one,
including when FP32 mass sums round below one (upstream
[#4204](https://github.com/radixark/miles/issues/4204) and proposed
[#4310](https://github.com/radixark/miles/pull/4310)). Partial candidate heads
retain the original approximate tail estimator. Core policy training requires
unfiltered rollout sampling until [#52](https://github.com/allenai/miles-olmo-core/issues/52) is resolved. The CPU tests
cover candidate preservation through Core packing and the complete-support
loss/gradient; a GPU training run is separate qualification.
