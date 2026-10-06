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
