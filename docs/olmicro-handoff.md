# Olmicro RL training — handoff to Jacob

**Working stack: native Olmicro inference in SGLang, RL orchestration in MILES, training in OLMo-core.** Validated with 500 updates of 450M GSM8K training on 4 B300s (2 training GPUs + 2 inference GPUs).

**Code — use these branches**

| Repository | Branch / contents |
| --- | --- |
| [Open-Instruct](https://github.com/allenai/open-instruct/tree/codex/olmicro-mainline-sync) | `codex/olmicro-mainline-sync`: config-driven launcher, runtime image definition and example run config |
| [miles-olmo-core](https://github.com/allenai/miles-olmo-core/tree/codex/olmicro-mainline-sync) | `codex/olmicro-mainline-sync`: MILES/OLMo-core integration; validated revision `5345776b1292d8b67149ceb47959a807bf1bd655` |
| Native SGLang changes | Stored in MILES at [`patches/sglang-olmicro-native/sglang.patch`](https://github.com/allenai/miles-olmo-core/blob/5345776b1292d8b67149ceb47959a807bf1bd655/patches/sglang-olmicro-native/sglang.patch). Applied by the image build to upstream SGLang `b84bec5761a6cdc8b1df86b9b89ead80a7a9bb02`; no separate published SGLang branch required. |

- All dependency revisions, including OLMo-core, are pinned in Open-Instruct's [`runtime/miles/runtime.lock.json`](https://github.com/allenai/open-instruct/blob/codex/olmicro-mainline-sync/runtime/miles/runtime.lock.json).
- Cluster working copies: `/weka/oe-adapt-default/michaeln/olmicro/{open-instruct,miles-olmo-core,sglang}`. Local copies: `/home/mnoukhov/olmicro/` under the same names. Use your own checkout for changes.

**Ready-to-use image**

- Beaker alias: **`michaeln/olmicro-miles-84848b6`**.
- Immutable launch ID: **`01M6HMYBFW8D9ZK4SKGJ1XD6HW`**.
- Contains the working stack; no dependency build or sibling checkouts are needed just to launch training. Image Open-Instruct revision: `84848b6ec4b46e8a0c16aa8a5df491ae35c8ce1a`.

**Starting a run**

- [Validated B300 example config](https://github.com/allenai/open-instruct/blob/codex/olmicro-mainline-sync/configs/miles/benchmarks/hybrid_450m_gsm8k_native_b300.toml).

```bash
git clone --branch codex/olmicro-mainline-sync https://github.com/allenai/open-instruct.git
cd open-instruct
mkdir -p runs
cp configs/miles/benchmarks/hybrid_450m_gsm8k_native_b300.toml runs/my-olmicro.toml
```

- This is the validated B300 config: BF16, Core FA4, native Triton/GDN inference with decode CUDA graphs. It contains Michael's paths/account settings: change run `name`, `output.root`, W&B settings and secret references to your own; confirm access to workspace/budget and input files.
- Change model, data and training parameters in that TOML for your experiment. Keep the validated backend settings initially. The launcher needs Python 3.12 and authenticated Beaker CLI; see [launching.md](https://github.com/allenai/open-instruct/blob/codex/olmicro-mainline-sync/docs/miles/launching.md).

```bash
python -m open_instruct.miles plan runs/my-olmicro.toml
python -m open_instruct.miles validate runs/my-olmicro.toml
export MILES_EXISTING_IMAGE=01M6HMYBFW8D9ZK4SKGJ1XD6HW
python -m open_instruct.miles run runs/my-olmicro.toml
```

- Keep tracked source clean/committed. The image supplies runtime code; local Python edits require a new image to take effect.
- Establish persistent monitoring ownership before launching. The previous monitor finished its run and does not automatically cover yours.

**Files on Weka**

| What | Location |
| --- | --- |
| Initial 450M HF model | `/weka/oe-training-default/yashasbls/hybridish-local/hf/v1/450m/midtrain/step0038147` |
| Matching Core model config | `/weka/oe-training-default/yashasbls/hybridish-local/olmo-core/v1/450m/midtrain/step0038147/config.json` |
| GSM8K prompts and reward config | `/weka/oe-adapt-default/michaeln/open-instruct-hybrid/data/gsm8k-base/` — `train.jsonl`, `eval.jsonl`, `verifiers.json` |
| Completed run | `/weka/oe-adapt-default/michaeln/open-instruct-hybrid/runs/hybrid-450m-mainline-b300-500-20261103/` |
| Final training checkpoint, relative to run | `checkpoints/core/rollout_0000499` — includes optimizer/RNG/data state |
| Final HF model, relative to run | `export-hf/` — model export, not full training-resume state |
| Run receipts and logs | `/weka/oe-adapt-default/michaeln/olmicro/monitor-b300/` |

The `open-instruct-hybrid` data/output paths are intentional; they were not moved with the code.

**Validation reference:** [successful run](https://beaker.org/ex/01M6HQTBPF1PVWNNWWCDYVYRJD), [W&B](https://wandb.ai/mnoukhov/open_instruct_internal/runs/xkprs8cy), [publication/graphs/resume smoke](https://beaker.org/ex/01M6HN0A01NC3HN9MG0V12Q6AN). This configuration works on B300s; numerical parity with vLLM remains unresolved, and other model sizes/configurations need their own training smoke.
