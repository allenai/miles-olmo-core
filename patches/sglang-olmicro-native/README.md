# Native Olmicro SGLang source overlay

This patch adds a native `MainlineLadderForCausalLM` to SGLang commit
`3145136dcd1238754e0ea2b2ffd546532119c71c`. It uses batched native GDN kernels,
SGLang-managed recurrent/convolution state and paged full attention. It does
not call a Transformers model forward. Release checkpoint and publication
parameter names are preserved; QKV, gate/up and convolution weights are packed after loading, including partial updates.

The supported configuration is BF16, TP1/PP1, FP32 recurrent state and GDN gate computation,
no quantization, no CUDA graphs, no prefix cache, and no speculative decoding.
The five final-midtrain sizes are tested on one Neptune L40S. The final
per-size outcomes, including failed gates, are recorded in [qualification.json](qualification.json).
Thirty-five focused CPU/GPU regressions passed together. Native generation,
queueing, slot reuse/reload, selected-token log probabilities, and full-vocabulary
KL are separate gates; a functional pass does not imply numerical equivalence.
The bundled patch is the shared source artifact. Local source provenance is
`986877f4cd38b090ce2338e685f914cd65c18e8c`; reproduce from the pinned upstream
commit and this patch rather than expecting that local commit upstream.

## Results

Native functional checks passed at all five sizes. **Strict vLLM numerical
compatibility is not fully qualified.** The KL gate is maximum bidirectional
full-vocabulary KL <= 0.001 across 50 fixed prefixes; the selected-token gate is
maximum absolute log-probability error <= 0.08.

| Size | Native functional | Max KL | KL gate | Selected-token gate |
|---|---|---:|---|---|
| 275m | pass | 0.000857 | pass | fail: 0.125 |
| 450m | pass | 0.000788 | pass | fail: 0.156 |
| 810m | pass | 0.001380 | fail | blocked by reference failure |
| 1.4b | pass | 0.000995 | pass | fail: 0.0956 |
| 2.7b | pass | 0.000759 | pass | blocked by reference failure |

The vLLM reference changed exact greedy outputs after slot reuse at 810m and
under queueing at 2.7b. Those failed checks are retained; the selected comparator
refuses to declare a pass for them. Cross-backend greedy continuations also
sometimes differ. Remaining numerical differences have not all been root-caused;
the engines use different pinned Torch/CUDA stacks (2.13/CUDA13 versus
2.10/CUDA12.8). Per-prefix [comparison reports](reports/) preserve all failures.
The final 2.7b comparison was computed locally from saved vectors after GPU
preemption. No tolerance was relaxed.

Apply to a clean SGLang checkout at that exact commit:

```bash
git checkout 3145136dcd1238754e0ea2b2ffd546532119c71c
git apply --check /path/to/miles-olmo-core/patches/sglang-olmicro-native/sglang.patch
git apply /path/to/miles-olmo-core/patches/sglang-olmicro-native/sglang.patch
```

Use the patched `python/` first on `PYTHONPATH` in the pinned SGLang environment,
or include the patch in a cluster image build. The full test commands and
standalone server flags are then in `test/manual/olmicro/README.md` in SGLang.
The patch includes the manual tests and comparison probes. Select full-attention
backend `fa3`, linear-attention backend `triton`, and
`SGLANG_USE_SGL_FA3_KERNEL=1`; the pinned native kernels support the L40S (SM89).
Triton and FlashInfer full-attention alternatives were investigated but are not
interchangeable with the qualified vLLM reference in the strict numerical checks.
Use the explicit flags in the patched manual-test README.

GPU sessions:
[initial](https://beaker.org/job/01M675BCQ9F5G11AP6DEAAMAAW),
[replacement](https://beaker.org/job/01M682TSK1EBFADEBFQB9XWGSG).
The initial session was preempted; the replacement uses the same native image
`01M607Z48PYPNN0KHVZ5PGYJEG`. The reference uses Core/vLLM image
`01M64G5M526S9HNF960NP3QRVB`. Both use one L40S, BF16 and FP32 recurrent state.
Raw final native reports are under
`/weka/oe-adapt-default/michaeln/olmicro/native-sglang-20261030/native-fa3-final/`;
reference reports and earlier candidates are in the parent directory.

Existing MILES benchmark recipes explicitly select the older HF reference
bridge. A future native training smoke must change that selection and verify
cross-process learner weight publication and optimizer updates. These serving
checks do not change a running experiment or qualify the complete RL path.
