# Third-party notices

miles-olmo-core Copyright © 2026 The Allen Institute for Artificial Intelligence

This fork is derived from [Miles](https://github.com/radixark/miles), which was forked from [slime](https://github.com/THUDM/slime). Original Miles and AllenAI code is licensed under Apache-2.0; see [LICENSE](LICENSE). Third-party material retains its respective copyrights and licenses.

## Scope

These notices cover third-party code adapted or included in this repository, including upstream context retained in patch files. They preserve the collected copyright notices, complete license texts, and upstream NOTICE material. File-level attribution comments remain in place.

User-installed dependency packages and their native libraries, prebuilt container images, model weights, and datasets are not distributed with this source release. The separate [dependency review list](DEPENDENCIES.md) labels those packages **not distributed**. A library can appear in both lists when an adapted portion or patch context is included here while the complete library is installed separately.

License evidence was collected on 2026-10-04; distribution scope was updated on 2026-10-05 against source snapshot `664f4a8c346828e8ee1d996859e2d373ce54d7f7`. A **license evidence revision** identifies the inspected license files, not necessarily the revision from which code was copied. Entries state when copied-source revisions are recorded or remain unknown.

## Included upstream and adapted source

### deepscaler

Source: [agentica-project/deepscaler](https://github.com/agentica-project/deepscaler). License: **MIT**.

miles/rollout/rm_hub/math_utils.py records this source revision.

License evidence revision: `e6080ccd974eb64bd3430f0b36108244a6fee330`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/deepscaler.txt).

### flash-attention-source

Source: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention). License: **BSD-3-Clause**.

Adapted Triton attention backward kernels in miles/backends/fsdp_utils/sglang_attn_bridge/triton_attn_bwd.py; original upstream revision is not recorded. FlashAttention is distinct from the removed FlashInfer-adapted fake-QAT kernel.

License evidence revision: `e9515d5dee6ade134a33d6020d38d01ef0596996`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/flash-attention-source.txt).

### lm-evaluation-harness

Source: [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). License: **MIT (upstream); Apache-2.0 notices retained in Miles adaptation**.

Adapted math utility in miles/rollout/rm_hub/math_dapo_utils.py; that file also retains Apache-2.0 notices for Bytedance and EleutherAI/HuggingFace. Original upstream revision is not recorded.

License evidence revision: `d6de81643928d653435c431bae19945d41d32520`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/lm-evaluation-harness.txt).

### mbridge

Source: [ISEEKYAN/mbridge](https://github.com/ISEEKYAN/mbridge). License: **BSD-3-Clause AND Apache-2.0 AND MIT**.

Retained adapter miles_plugins/mbridge/mimo.py has an NVIDIA copyright header and uses mbridge. These related upstream notices are retained conservatively. The copied-source provenance of the adapter is not recorded; the pinned mbridge license revision does not establish that provenance.

License evidence revision: `89eb10887887bc74853f89a4de258c0702932a1c`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/mbridge.txt).

### megatron-bridge

Source: [radixark/Megatron-Bridge](https://github.com/radixark/Megatron-Bridge). License: **Apache-2.0**.

docker/npu_patch/megatron_bridge.patch; external library. CUDA recipe pins this revision; NPU guide uses floating bridge branch.

License evidence revision: `2e09c234a3272285140224d0d698593418b55ba5`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/megatron-bridge.txt).

### megatron-lm-fork

Source: [radixark/Megatron-LM](https://github.com/radixark/Megatron-LM). License: **BSD-3-Clause; Apache-2.0 and MIT portions**.

Adapted miles_plugins/optimizers/nvme_stream.py records this revision. Retained AMD/NPU patch files target external Megatron installs; the NPU guide selects 3714d81d418c9f1bca4594fc35f9e8289f652862.

License evidence revision: `4716f75475c78e2fc2c6f0d3af095f1681b770b4`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/megatron-lm-fork.txt).

### megatron-lm-source

Source: [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM). License: **BSD-3-Clause; Apache-2.0 and MIT portions**.

Adapted model_provider.py records this revision; scheduler, checkpoint and optimizer helpers also retain Megatron-derived code. Original revisions for all such portions are not recorded.

License evidence revision: `b1efb3c7126ef7615e8c333432d76e08038e17ff`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/megatron-lm-source.txt).

### miles-upstream

Source: [radixark/miles](https://github.com/radixark/miles). License: **Apache-2.0**.

Upstream Miles code, modified by AllenAI. Upstream reference dbbab1566ae438f7202fff653eae938e07b1d4b6; release source snapshot 664f4a8c346828e8ee1d996859e2d373ce54d7f7. Copyright notices in individual files remain in place, including NVIDIA and Bytedance/EleutherAI/HuggingFace notices.

License evidence revision: `dbbab1566ae438f7202fff653eae938e07b1d4b6`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/miles-upstream.txt).

### openrlhf

Source: [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). License: **Apache-2.0**.

Adapted miles/ray/utils.py and miles/backends/training_utils/loss_hub/math_utils.py; revision recorded in those files.

License evidence revision: `10c733694ed9fbb78a0a2ff6a05efc7401584d46`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/openrlhf.txt).

### pai-megatron-patch

Source: [alibaba/Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch). License: **Apache-2.0**.

Adapted tools/fp8_cast_bf16.py records this revision.

License evidence revision: `2b201af08336dea0403df7c6b497c964cf5a2e75`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/pai-megatron-patch.txt).

### pytorch

Source: [pytorch/pytorch](https://github.com/pytorch/pytorch). License: **BSD-3-Clause**.

Adapted gather_object in miles/utils/ft_utils/process_group_utils.py records v2.11.0; miles/utils/distributed_utils.py also references PyTorch.

License evidence revision: `70d99e998b4955e0049d13a98d77ae1b14db1f45`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/pytorch.txt).

### ray-source

Source: [ray-project/ray](https://github.com/ray-project/ray). License: **Apache-2.0**.

Accelerator environment-variable references in miles/ray/utils.py record this revision.

License evidence revision: `161849364a784442cc659fb9780f1a6adee85fce`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/ray-source.txt).

### sglang-source

Source: [sgl-project/sglang](https://github.com/sgl-project/sglang). License: **Apache-2.0**.

Adapted process helper in miles/utils/external_utils/command_utils.py and docker/npu_patch/sglang.patch. This is a reference-runtime license revision; copied-source revisions are not recorded. DeepSeek-V3.2 encoder was removed.

License evidence revision: `3145136dcd1238754e0ea2b2ffd546532119c71c`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/sglang-source.txt).

### slime

Source: [THUDM/slime](https://github.com/THUDM/slime). License: **Apache-2.0**.

Ancestor of Miles, acknowledged in README.md; original base revision is not recorded here.

License evidence revision: `8c17b676cb57af1d17ee4402e91e9209af84b60b`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/slime.txt).

### torchft-source

Source: [pytorch/torchft](https://github.com/pytorch/torchft). License: **BSD-3-Clause**.

miles/utils/test_utils/fault_injector.py references examples/monarch/utils/failure.py; original upstream revision is not recorded.

License evidence revision: `14c8c6bcfbd6e0f009ff7901ba933cc12f477376`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/torchft-source.txt).

### transformer-engine

Source: [NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine). License: **Apache-2.0**.

docker/patch/cu13/*.patch contain modifications and upstream context for external TransformerEngine 2.17.0; library is installed separately.

License evidence revision: `2e559f062497bef768dfbe9d7e45548fadeca80a`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/transformer-engine.txt).

### transformers-source

Source: [huggingface/transformers](https://github.com/huggingface/transformers). License: **Apache-2.0**.

Adapted Qwen gated delta-net modules in miles_plugins/models/qwen3_next.py and qwen3_5.py; recorded revision in qwen3_next.py.

License evidence revision: `38a08b6e8ae35857109cedad75377997fecbf9d0`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/transformers-source.txt).

### triton-source

Source: [triton-lang/triton](https://github.com/triton-lang/triton). License: **MIT**.

Same Triton attention backward file credits tutorial 06-fused-attention; original upstream revision is not recorded.

License evidence revision: `6fe3348e6e907584e07e593ab19a35caee95c37a`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/triton-source.txt).

### typer-source

Source: [fastapi/typer](https://github.com/fastapi/typer). License: **MIT (upstream Typer); credited issue-comment license not established**.

miles/utils/typer_utils.py credits issue 154 comment 1544876144. The comment does not record an upstream revision; repository license collected separately. The credited comment is by tbenthompson (https://github.com/fastapi/typer/issues/154#issuecomment-1544876144). The issue comment has no explicit license statement; the repository MIT license is evidence for Typer, not confirmation of a separate grant for the comment.

License evidence revision: `a80f6e5ecd74f32b983cca336a2f3cba98d9853a`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/typer-source.txt).

### verl

Source: [verl-project/verl](https://github.com/verl-project/verl). License: **Apache-2.0**.

Adapted miles/utils/seqlen_balancing.py. Additional recorded revisions: c3b20575d2bc815fcccd84bddb4c0401fc4b632b (model_provider.py), 0bdf7f469854815177e73dcfe9e420836c952e6e (loss_hub/math_utils.py). FSDP helper references do not record a revision.

License evidence revision: `468adf22c43b744348051fccd7a5d830c6c3c36a`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/verl.txt).

## Maintenance and provenance notes

When adding or modifying included third-party code, preserve its copyright, LICENSE, and NOTICE material and update these entries. The [machine-readable attribution inventory](third_party/inventory.json) records source links, license evidence revisions, and SHA-256 checksums for every retained local notice file. Update the separate dependency review list when changing user-installed dependencies.

The ProRL-Agent-Server streaming adaptation, DeepSeek-V3.2 encoder, FlashInfer-adapted fake-QAT kernel, DeepSeek-V4 conversion helper, MindSpeed source patch, and TileKernels/TileLang source patches are absent from this release. The dependency review list distinguishes those removals from external libraries still referenced by installation recipes.

The Typer issue-comment attribution has no explicit license statement in the credited comment. Its credit is preserved, but the Typer repository MIT license does not establish a separate grant for the comment. Several inherited adaptations lack recorded copied-source revisions, and the mbridge adapter has unresolved copied-source provenance; their entries distinguish those limitations from the available license evidence.
