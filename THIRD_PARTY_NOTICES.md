# Third-party notices

miles-olmo-core Copyright © 2026 The Allen Institute for Artificial Intelligence

This fork is derived from [Miles](https://github.com/radixark/miles), which was forked from [slime](https://github.com/THUDM/slime). Original Miles and AllenAI code is licensed under Apache-2.0; see [LICENSE](LICENSE). Third-party material retains its respective copyrights and licenses.

## Scope

This is a source-code release. The first section identifies third-party code or patch context included in the repository. The remaining sections document separately installed dependencies, including optional backends and inherited build recipes. Dependency packages, prebuilt container images, model weights, and datasets are not bundled in this release. An entry does not imply that a particular backend is supported by this fork.

The linked local files are part of this notice: they preserve the collected copyright notices, complete license texts, and upstream NOTICE material. Each component’s text file groups its collected documents under their original filenames. Upstream notices can mention subcomponents of an external package; those mentions do not mean those subcomponents are included in Miles.

This inventory covers MIT, BSD (with clause variants), ISC, and Apache-2.0 components, including mixed-license packages that contain these licenses. Complete mixed-license notices are preserved. Other license families and proprietary prerequisites are outside this inventory’s scope; omission is not a statement of approval or non-use.

Inventory date: 2026-10-04 UTC. The source snapshot is `5396339f1f118f544e5a2fdd0c5cb0677739ff9b`, with `tools/convert_mxfp4_to_fp8.py` removed. Python versions reflect a fresh Python 3.12 resolution of the base requirements and declared extras, supplemented by optional-serving versions recorded in the dependency review. These are inspected versions, not new installation pins. Build recipes with floating branches or wheel selection can resolve differently.

For source entries, a **license evidence revision** identifies the inspected license files. It is not a claim that code was copied from that revision unless the entry explicitly says the revision is recorded in the retained source. File-level attribution comments remain in place.

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

Upstream Miles code, modified by AllenAI. Upstream reference dbbab1566ae438f7202fff653eae938e07b1d4b6; release snapshot 5396339f1f118f544e5a2fdd0c5cb0677739ff9b. Copyright notices in individual files remain in place, including NVIDIA and Bytedance/EleutherAI/HuggingFace notices.

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

Source: [fastapi/typer](https://github.com/fastapi/typer). License: **MIT**.

miles/utils/typer_utils.py credits issue 154 comment 1544876144. The comment does not record an upstream revision; repository license collected separately. The credited comment is by tbenthompson (https://github.com/fastapi/typer/issues/154#issuecomment-1544876144). The issue comment has no explicit license statement; the repository MIT license is evidence for Typer, not confirmation of a separate grant for the comment.

License evidence revision: `a80f6e5ecd74f32b983cca336a2f3cba98d9853a`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/typer-source.txt).

### verl

Source: [verl-project/verl](https://github.com/verl-project/verl). License: **Apache-2.0**.

Adapted miles/utils/seqlen_balancing.py. Additional recorded revisions: c3b20575d2bc815fcccd84bddb4c0401fc4b632b (model_provider.py), 0bdf7f469854815177e73dcfe9e420836c952e6e (loss_hub/math_utils.py). FSDP helper references do not record a revision.

License evidence revision: `468adf22c43b744348051fccd7a5d830c6c3c36a`.

Copyright, license and notice files: [Full license and notices](third_party/licenses/verl.txt).

## Separately installed source and build dependencies

These libraries and tools are obtained separately. Any retained patches or adapted portions are identified in the preceding section. Source-license evidence does not describe every binary component in an externally built environment.

| Component | Recipe/reference version | License | Source and full notices |
| --- | --- | --- | --- |
| apex | master | BSD-3-Clause | [source](https://github.com/NVIDIA/apex/tree/575968bc1f9127ccd61003a681472a83af4ff1a1); [Full license and notices](third_party/licenses/apex.txt) |
| causal-conv1d | v1.6.1 | BSD-3-Clause | [source](https://github.com/Dao-AILab/causal-conv1d/tree/c51519fb10d92ffa257c1982e1831d3243454b2f); [Full license and notices](third_party/licenses/causal-conv1d.txt) |
| emerging-optimizers | v0.3.0 | Apache-2.0 | [source](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/tree/b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16); [Full license and notices](third_party/licenses/emerging-optimizers.txt) |
| fast-hadamard-transform | e7706faf8d1c3b9f241e36860640ad1dac644ede | BSD-3-Clause | [source](https://github.com/Dao-AILab/fast-hadamard-transform/tree/e7706faf8d1c3b9f241e36860640ad1dac644ede); [Full license and notices](third_party/licenses/fast-hadamard-transform.txt) |
| flash-linear-attention | v0.5.2 | MIT | [source](https://github.com/fla-org/flash-linear-attention/tree/9c8e42e762fce087c27b673af4922795d9edb85e); [Full license and notices](third_party/licenses/flash-linear-attention.txt) |
| flash-mla | main | MIT | [source](https://github.com/deepseek-ai/FlashMLA/tree/2e5429fc5653bab6e081f09477126f731882a6a9); [Full license and notices](third_party/licenses/flash-mla.txt) |
| flashqla | 7c7dfe16416ad21b1d03258189fc8d3b8460ae06 | MIT | [source](https://github.com/QwenLM/FlashQLA/tree/7c7dfe16416ad21b1d03258189fc8d3b8460ae06); [Full license and notices](third_party/licenses/flashqla.txt) |
| ifbench | main | Apache-2.0 | [source](https://github.com/allenai/IFBench/tree/1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d); [Full license and notices](third_party/licenses/ifbench.txt) |
| mamba-ssm | v2.3.1 | Apache-2.0 | [source](https://github.com/state-spaces/mamba/tree/c5afbdf3bda1a09d68f65181ae3a43ec71079820); [Full license and notices](third_party/licenses/mamba-ssm.txt) |
| mbridge | 89eb10887887bc74853f89a4de258c0702932a1c | BSD-3-Clause AND Apache-2.0 AND MIT | [source](https://github.com/ISEEKYAN/mbridge/tree/89eb10887887bc74853f89a4de258c0702932a1c); [Full license and notices](third_party/licenses/mbridge.txt) |
| megatron-energon | main | BSD-3-Clause | [source](https://github.com/NVIDIA/Megatron-Energon/tree/0397ac7f653468851dc0991525c7bc4a51d58d8b); [Full license and notices](third_party/licenses/megatron-energon.txt) |
| mooncake | 0.3.12.post1 (ROCm recipe); wheel-selected CUDA variant | Apache-2.0 | [source](https://github.com/kvcache-ai/Mooncake/tree/0d1a8040faebb7c127c8901840a38c2ff57e80c5); [Full license and notices](third_party/licenses/mooncake.txt) |
| multi-storage-client | main | Apache-2.0 | [source](https://github.com/NVIDIA/multi-storage-client/tree/4188de82250d0d56c36895cbc464bcf7f4eca842); [Full license and notices](third_party/licenses/multi-storage-client.txt) |
| nccl-tests | ae98985f5599617be94042f4aa3637d10014ce89 | BSD-3-Clause | [source](https://github.com/NVIDIA/nccl-tests/tree/ae98985f5599617be94042f4aa3637d10014ce89); [Full license and notices](third_party/licenses/nccl-tests.txt) |
| nvdlfw-inspect | main | Apache-2.0 | [source](https://github.com/NVIDIA/nvidia-dlfw-inspect/tree/4118044cc84f0183714a2ab1bc215fa49f9aaa82); [Full license and notices](third_party/licenses/nvdlfw-inspect.txt) |
| nvidia-modelopt | main | Apache-2.0 | [source](https://github.com/NVIDIA/Model-Optimizer/tree/6a1068746dd0e746fe4bcb79152b86bdcf58cb41); [Full license and notices](third_party/licenses/nvidia-modelopt.txt) |
| olmo-core-runtime | 4f00ddcb475c3c9dc6c7005217b7572b5bf7582f | Apache-2.0 with third-party notices | [source](https://github.com/allenai/OLMo-core/tree/4f00ddcb475c3c9dc6c7005217b7572b5bf7582f); [Full license and notices](third_party/licenses/olmo-core-runtime.txt) |
| olmo-sglang-runtime | b3a795f33e048dac401c89bb5823381146181ac6 | Apache-2.0 with third-party notices | [source](https://github.com/allenai/olmo-sglang/tree/b3a795f33e048dac401c89bb5823381146181ac6); [Full license and notices](third_party/licenses/olmo-sglang-runtime.txt) |
| open-instruct | c20ab00a8175b35517629b14ea2404b0be924ac6 | Apache-2.0 | [source](https://github.com/allenai/open-instruct/tree/c20ab00a8175b35517629b14ea2404b0be924ac6); [Full license and notices](third_party/licenses/open-instruct.txt) |
| rocm-systems | 746c7b3c9e78b094389e19499919fdd43a0b6a90 | BSD-3-Clause | [source](https://github.com/ROCm/rocm-systems/tree/746c7b3c9e78b094389e19499919fdd43a0b6a90); [Full license and notices](third_party/licenses/rocm-systems.txt) |
| sgl-kernel-npu | 2026.05.01 | MIT | [source](https://github.com/sgl-project/sgl-kernel-npu/tree/775c4204fcc12e97b22610a62a4a355348c70a90); [Full license and notices](third_party/licenses/sgl-kernel-npu.txt) |
| sgl-router-for-miles | main | Apache-2.0 | [source](https://github.com/radixark/sgl-router-for-miles/tree/3145136dcd1238754e0ea2b2ffd546532119c71c); [Full license and notices](third_party/licenses/sgl-router-for-miles.txt) |
| tilelang-cuda | 7e3bbf3703e769479757eb7ecec2f2626f3cd0a8 | MIT with additional historical collaboration note | [source](https://github.com/tile-ai/tilelang/tree/7e3bbf3703e769479757eb7ecec2f2626f3cd0a8); [Full license and notices](third_party/licenses/tilelang-cuda.txt) |
| torch-memory-saver-cuda | b5588e83de86412a48689a6583a4b567e75f7acc | MIT | [source](https://github.com/fzyzcjy/torch_memory_saver/tree/b5588e83de86412a48689a6583a4b567e75f7acc); [Full license and notices](third_party/licenses/torch-memory-saver-cuda.txt) |
| torch-memory-saver-rocm | 06caa534822c5b980b61733a5d79ac731f9a5f9c | MIT | [source](https://github.com/fzyzcjy/torch_memory_saver/tree/06caa534822c5b980b61733a5d79ac731f9a5f9c); [Full license and notices](third_party/licenses/torch-memory-saver-rocm.txt) |
| torch-npu | c4194988d3c830375abf08e01d5bec3fe6a86406 | BSD-3-Clause | [source](https://github.com/Ascend/pytorch/tree/c4194988d3c830375abf08e01d5bec3fe6a86406); [Full license and notices](third_party/licenses/torch-npu.txt) |
| transformer-engine-rocm | a365f2deb555e681bf5b259e344b2bd1fbb4f87c | Apache-2.0 AND MIT | [source](https://github.com/ROCm/TransformerEngine/tree/a365f2deb555e681bf5b259e344b2bd1fbb4f87c); [Full license and notices](third_party/licenses/transformer-engine-rocm.txt) |

TileLang’s optional CUDA recipe still selects 0.1.14. Its original LICENSE and THIRDPARTYNOTICES text are included without removing the historical Microsoft collaboration note. Providing these notices does not resolve the applicability of those additional terms.

## Separately installed Python dependencies

“Base/extras” means the current requirements or declared optional extras, including transitive dependencies. “Serving reference” means an optional-serving version from the dependency inventory. “Optional integration” identifies retained Tinker or IFBench usage. Package archives and installed distribution metadata provide the notices; when an archive omits them, the linked upstream files and their immutable evidence revisions are recorded in [the inventory](third_party/inventory.json).

| Package | Inspected version | License | Inventory scope | Source and full notices |
| --- | --- | --- | --- | --- |
| absl-py | 2.5.0 | Apache-2.0 | Base/extras | [source](https://github.com/abseil/abseil-py); [Full license and notices](third_party/licenses/absl-py-2-5-0.txt) |
| accelerate | 1.15.0 | Apache-2.0 | Base/extras | [source](https://github.com/huggingface/accelerate); [Full license and notices](third_party/licenses/accelerate-1-15-0.txt) |
| ai2-olmo-core | 3.0.0 | Apache-2.0 | Serving reference | [source](https://github.com/allenai/Olmo-core); [Full license and notices](third_party/licenses/ai2-olmo-core-3-0-0.txt) |
| aiohttp | 3.14.3 | Apache-2.0 AND MIT | Base/extras | [source](https://github.com/aio-libs/aiohttp); [Full license and notices](third_party/licenses/aiohttp-3-14-3.txt) |
| aiohttp-cors | 0.8.1 | Apache-2.0 | Base/extras | [source](https://github.com/aio-libs/aiohttp-cors); [Full license and notices](third_party/licenses/aiohttp-cors-0-8-1.txt) |
| aiosignal | 1.4.0 | Apache-2.0 | Base/extras | [source](https://github.com/aio-libs/aiosignal); [Full license and notices](third_party/licenses/aiosignal-1-4-0.txt) |
| airportsdata | 20260905 | MIT | Serving reference | [source](https://github.com/mborsetti/airportsdata/); [Full license and notices](third_party/licenses/airportsdata-20260905.txt) |
| alembic | 1.20.0 | MIT | Base/extras | [source](https://github.com/sqlalchemy/alembic/); [Full license and notices](third_party/licenses/alembic-1-20-0.txt) |
| annotated-doc | 0.0.5 | MIT | Base/extras | [source](https://github.com/fastapi/annotated-doc); [Full license and notices](third_party/licenses/annotated-doc-0-0-5.txt) |
| annotated-types | 0.8.0 | MIT | Base/extras | [source](https://github.com/annotated-types/annotated-types); [Full license and notices](third_party/licenses/annotated-types-0-8-0.txt) |
| antlr4-python3-runtime | 4.9.3 | BSD-3-Clause | Base/extras | [source](http://www.antlr.org); [Full license and notices](third_party/licenses/antlr4-python3-runtime-4-9-3.txt) |
| anyio | 4.15.1 | MIT | Base/extras | [source](https://pypi.org/project/anyio/4.15.1/); [Full license and notices](third_party/licenses/anyio-4-15-1.txt) |
| apache-tvm-ffi | 0.1.11 | Apache-2.0 | Serving reference | [source](https://github.com/apache/tvm-ffi); [Full license and notices](third_party/licenses/apache-tvm-ffi-0-1-11.txt) |
| attrs | 26.1.0 | MIT | Base/extras | [source](https://pypi.org/project/attrs/26.1.0/); [Full license and notices](third_party/licenses/attrs-26-1-0.txt) |
| av | 19.0.1 | BSD-3-Clause | Base/extras | [source](https://github.com/PyAV-Org/PyAV); [Full license and notices](third_party/licenses/av-19-0-1.txt) |
| blake3 | 1.0.9 | CC0-1.0 OR Apache-2.0 | Base/extras | [source](https://github.com/oconnor663/blake3-py); [Full license and notices](third_party/licenses/blake3-1-0-9.txt) |
| blinker | 1.9.0 | MIT | Base/extras | [source](https://github.com/pallets-eco/blinker/); [Full license and notices](third_party/licenses/blinker-1-9-0.txt) |
| bracex | 3.0.1 | MIT | Base/extras | [source](https://github.com/facelessuser/bracex); [Full license and notices](third_party/licenses/bracex-3-0-1.txt) |
| cachetools | 7.2.0 | MIT | Base/extras | [source](https://github.com/tkem/cachetools/); [Full license and notices](third_party/licenses/cachetools-7-2-0.txt) |
| cbor2 | 6.1.5 | MIT | Base/extras | [source](https://github.com/agronholm/cbor2); [Full license and notices](third_party/licenses/cbor2-6-1-5.txt) |
| cffi | 2.1.1 | MIT-0 | Base/extras | [source](https://github.com/python-cffi/cffi); [Full license and notices](third_party/licenses/cffi-2-1-1.txt) |
| charset-normalizer | 3.5.2 | MIT | Base/extras | [source](https://pypi.org/project/charset-normalizer/3.5.2/); [Full license and notices](third_party/licenses/charset-normalizer-3-5-2.txt) |
| chz | 0.4.0 | MIT | Optional integration | [source](https://github.com/openai/chz); [Full license and notices](third_party/licenses/chz-0-4-0.txt) |
| click | 8.5.0 | BSD-3-Clause | Base/extras | [source](https://github.com/pallets/click/); [Full license and notices](third_party/licenses/click-8-5-0.txt) |
| cloudpickle | 3.1.2 | BSD-3-Clause | Base/extras | [source](https://github.com/cloudpipe/cloudpickle); [Full license and notices](third_party/licenses/cloudpickle-3-1-2.txt) |
| colorful | 0.5.8 | MIT | Base/extras | [source](http://github.com/timofurrer/colorful); [Full license and notices](third_party/licenses/colorful-0-5-8.txt) |
| compressed-tensors | 0.19.0 | Apache-2.0 | Serving reference | [source](https://github.com/vllm-project/compressed-tensors); [Full license and notices](third_party/licenses/compressed-tensors-0-19-0.txt) |
| connectrpc | 0.11.1 | Apache-2.0 | Base/extras | [source](https://github.com/connectrpc/connect-py); [Full license and notices](third_party/licenses/connectrpc-0-11-1.txt) |
| contourpy | 1.4.0 | BSD-3-Clause | Base/extras | [source](https://github.com/contourpy/contourpy); [Full license and notices](third_party/licenses/contourpy-1-4-0.txt) |
| cryptography | 50.0.2 | Apache-2.0 OR BSD-3-Clause | Base/extras | [source](https://pypi.org/project/cryptography/50.0.2/); [Full license and notices](third_party/licenses/cryptography-50-0-2.txt) |
| cuda-bindings | 13.4.3 | Apache-2.0 | Base/extras | [source](https://github.com/NVIDIA/cuda-python); [Full license and notices](third_party/licenses/cuda-bindings-13-4-3.txt) |
| cuda-core | 1.2.1 | Apache-2.0 | Serving reference | [source](https://github.com/NVIDIA/cuda-python/tree/main/cuda_core); [Full license and notices](third_party/licenses/cuda-core-1-2-1.txt) |
| cuda-pathfinder | 1.8.3 | Apache-2.0 | Base/extras | [source](https://github.com/NVIDIA/cuda-python); [Full license and notices](third_party/licenses/cuda-pathfinder-1-8-3.txt) |
| cuda-python | 13.4.1 | Apache-2.0 | Serving reference | [source](https://github.com/NVIDIA/cuda-python/); [Full license and notices](third_party/licenses/cuda-python-13-4-1.txt) |
| cuda-tile | 1.6.0rc5 | Apache-2.0 | Serving reference | [source](https://github.com/nvidia/cutile-python); [Full license and notices](third_party/licenses/cuda-tile-1-6-0rc5.txt) |
| databricks-sdk | 0.67.0 | Apache-2.0 | Base/extras | [source](https://pypi.org/project/databricks-sdk/0.67.0/); [Full license and notices](third_party/licenses/databricks-sdk-0-67-0.txt) |
| dataclass-extensions | 0.5.0 | Apache-2.0 | Serving reference | [source](https://github.com/epwalsh/dataclass-extensions); [Full license and notices](third_party/licenses/dataclass-extensions-0-5-0.txt) |
| datasets | 5.0.1 | Apache-2.0 | Base/extras | [source](https://github.com/huggingface/datasets); [Full license and notices](third_party/licenses/datasets-5-0-1.txt) |
| dill | 0.4.1 | BSD-3-Clause | Base/extras | [source](https://github.com/uqfoundation/dill); [Full license and notices](third_party/licenses/dill-0-4-1.txt) |
| docker | 7.2.0 | Apache-2.0 | Base/extras | [source](https://github.com/docker/docker-py); [Full license and notices](third_party/licenses/docker-7-2-0.txt) |
| dockerfile-parse | 2.0.1 | BSD-3-Clause | Base/extras | [source](https://github.com/containerbuildsystem/dockerfile-parse); [Full license and notices](third_party/licenses/dockerfile-parse-2-0-1.txt) |
| e2b | 2.52.0 | MIT | Base/extras | [source](https://github.com/e2b-dev/e2b/tree/main/packages/python-sdk); [Full license and notices](third_party/licenses/e2b-2-52-0.txt) |
| emoji | 2.16.0 | BSD-3-Clause | Optional integration | [source](https://github.com/carpedm20/emoji/); [Full license and notices](third_party/licenses/emoji-2-16-0.txt) |
| fastapi | 0.142.2 | MIT | Base/extras | [source](https://github.com/fastapi/fastapi); [Full license and notices](third_party/licenses/fastapi-0-142-2.txt) |
| filelock | 4.0.10 | MIT | Base/extras | [source](https://github.com/tox-dev/py-filelock); [Full license and notices](third_party/licenses/filelock-4-0-10.txt) |
| fla-core | 0.5.2 | MIT | Serving reference | [source](https://github.com/fla-org/flash-linear-attention); [Full license and notices](third_party/licenses/fla-core-0-5-2.txt) |
| flash-attn-4 | 4.0.0b19 | BSD-3-Clause | Serving reference | [source](https://github.com/Dao-AILab/flash-attention); [Full license and notices](third_party/licenses/flash-attn-4-4-0-0b19.txt) |
| flashinfer-python | 0.6.17 | Apache-2.0 | Serving reference | [source](https://github.com/flashinfer-ai/flashinfer); [Full license and notices](third_party/licenses/flashinfer-python-0-6-17.txt) |
| Flask | 3.1.3 | BSD-3-Clause | Base/extras | [source](https://github.com/pallets/flask/); [Full license and notices](third_party/licenses/flask-3-1-3.txt) |
| flask-cors | 6.0.5 | MIT | Base/extras | [source](https://github.com/corydolphin/flask-cors); [Full license and notices](third_party/licenses/flask-cors-6-0-5.txt) |
| fonttools | 4.66.1 | MIT | Base/extras | [source](http://github.com/fonttools/fonttools); [Full license and notices](third_party/licenses/fonttools-4-66-1.txt) |
| frozenlist | 1.8.0 | Apache-2.0 | Base/extras | [source](https://github.com/aio-libs/frozenlist); [Full license and notices](third_party/licenses/frozenlist-1-8-0.txt) |
| fsspec | 2026.6.0 | BSD-3-Clause | Base/extras | [source](https://github.com/fsspec/filesystem_spec); [Full license and notices](third_party/licenses/fsspec-2026-6-0.txt) |
| gguf | 0.19.0 | MIT | Serving reference | [source](https://github.com/ggml-org/llama.cpp); [Full license and notices](third_party/licenses/gguf-0-19-0.txt) |
| gitdb | 4.0.12 | BSD-3-Clause | Base/extras | [source](https://github.com/gitpython-developers/gitdb); [Full license and notices](third_party/licenses/gitdb-4-0-12.txt) |
| GitPython | 3.2.0 | BSD-3-Clause | Base/extras | [source](https://pypi.org/project/GitPython/3.2.0/); [Full license and notices](third_party/licenses/gitpython-3-2-0.txt) |
| google-api-core | 2.40.0 | Apache-2.0 | Base/extras | [source](https://github.com/googleapis/google-cloud-python); [Full license and notices](third_party/licenses/google-api-core-2-40-0.txt) |
| google-auth | 2.59.1 | Apache-2.0 | Base/extras | [source](https://github.com/googleapis/google-cloud-python/tree/main/packages/google-auth); [Full license and notices](third_party/licenses/google-auth-2-59-1.txt) |
| googleapis-common-protos | 1.75.5 | Apache-2.0 | Base/extras | [source](https://github.com/googleapis/google-cloud-python/tree/main/packages/googleapis-common-protos); [Full license and notices](third_party/licenses/googleapis-common-protos-1-75-5.txt) |
| graphene | 3.4.3 | MIT | Base/extras | [source](https://github.com/graphql-python/graphene); [Full license and notices](third_party/licenses/graphene-3-4-3.txt) |
| graphql-core | 3.2.13 | MIT | Base/extras | [source](https://github.com/graphql-python/graphql-core); [Full license and notices](third_party/licenses/graphql-core-3-2-13.txt) |
| graphql-relay | 3.2.0 | MIT | Base/extras | [source](https://github.com/graphql-python/graphql-relay-py); [Full license and notices](third_party/licenses/graphql-relay-3-2-0.txt) |
| grpcio | 1.84.0 | Apache-2.0 | Base/extras | [source](https://github.com/grpc/grpc); [Full license and notices](third_party/licenses/grpcio-1-84-0.txt) |
| grpcio-health-checking | 1.81.1 | Apache-2.0 | Serving reference | [source](https://grpc.io); [Full license and notices](third_party/licenses/grpcio-health-checking-1-81-1.txt) |
| grpcio-reflection | 1.81.1 | Apache-2.0 | Serving reference | [source](https://grpc.io); [Full license and notices](third_party/licenses/grpcio-reflection-1-81-1.txt) |
| grpcio-tools | 1.84.0 | Apache-2.0 | Base/extras | [source](https://github.com/grpc/grpc/tree/master/tools/distrib/python/grpcio_tools); [Full license and notices](third_party/licenses/grpcio-tools-1-84-0.txt) |
| grpclib | 0.4.9 | BSD-3-Clause | Base/extras | [source](https://github.com/vmagamedov/grpclib); [Full license and notices](third_party/licenses/grpclib-0-4-9.txt) |
| gunicorn | 26.2.0 | MIT | Base/extras | [source](https://gunicorn.org); [Full license and notices](third_party/licenses/gunicorn-26-2-0.txt) |
| h11 | 0.16.0 | MIT | Base/extras | [source](https://github.com/python-hyper/h11); [Full license and notices](third_party/licenses/h11-0-16-0.txt) |
| h2 | 4.4.1 | MIT | Base/extras | [source](https://github.com/python-hyper/h2/); [Full license and notices](third_party/licenses/h2-4-4-1.txt) |
| hf-xet | 1.6.0 | Apache-2.0 | Base/extras | [source](https://github.com/huggingface/xet-core.git); [Full license and notices](third_party/licenses/hf-xet-1-6-0.txt) |
| hpack | 4.2.0 | MIT | Base/extras | [source](https://github.com/python-hyper/hpack/); [Full license and notices](third_party/licenses/hpack-4-2-0.txt) |
| httpcore | 1.0.9 | BSD-3-Clause | Base/extras | [source](https://github.com/encode/httpcore); [Full license and notices](third_party/licenses/httpcore-1-0-9.txt) |
| httpcore2 | 2.13.1 | BSD-3-Clause | Base/extras | [source](https://github.com/pydantic/httpx2/blob/main/src/httpcore2); [Full license and notices](third_party/licenses/httpcore2-2-13-1.txt) |
| httpx | 0.28.1 | BSD-3-Clause | Base/extras | [source](https://github.com/encode/httpx); [Full license and notices](third_party/licenses/httpx-0-28-1.txt) |
| httpx2 | 2.13.1 | BSD-3-Clause | Base/extras | [source](https://github.com/pydantic/httpx2); [Full license and notices](third_party/licenses/httpx2-2-13-1.txt) |
| huey | 3.4.0 | MIT | Base/extras | [source](https://github.com/coleifer/huey); [Full license and notices](third_party/licenses/huey-3-4-0.txt) |
| huggingface_hub | 1.33.0 | Apache-2.0 | Base/extras | [source](https://github.com/huggingface/huggingface_hub); [Full license and notices](third_party/licenses/huggingface-hub-1-33-0.txt) |
| humming-kernels | 0.1.10 | Apache-2.0 | Serving reference | [source](https://pypi.org/project/humming-kernels/0.1.10/); [Full license and notices](third_party/licenses/humming-kernels-0-1-10.txt) |
| hyperframe | 6.1.0 | MIT | Base/extras | [source](https://github.com/python-hyper/hyperframe/); [Full license and notices](third_party/licenses/hyperframe-6-1-0.txt) |
| idna | 3.20 | BSD-3-Clause | Base/extras | [source](https://github.com/kjd/idna); [Full license and notices](third_party/licenses/idna-3-20.txt) |
| importlib_metadata | 9.0.1 | Apache-2.0 | Base/extras | [source](https://github.com/python/importlib_metadata); [Full license and notices](third_party/licenses/importlib-metadata-9-0-1.txt) |
| iniconfig | 2.3.0 | MIT | Base/extras | [source](https://github.com/pytest-dev/iniconfig); [Full license and notices](third_party/licenses/iniconfig-2-3-0.txt) |
| interegular | 0.3.3 | MIT | Serving reference | [source](https://github.com/MegaIng/regex_intersections); [Full license and notices](third_party/licenses/interegular-0-3-3.txt) |
| ipython_pygments_lexers | 1.1.1 | BSD-3-Clause | Serving reference | [source](https://github.com/ipython/ipython-pygments-lexers); [Full license and notices](third_party/licenses/ipython-pygments-lexers-1-1-1.txt) |
| itsdangerous | 2.2.0 | BSD-3-Clause | Base/extras | [source](https://github.com/pallets/itsdangerous/); [Full license and notices](third_party/licenses/itsdangerous-2-2-0.txt) |
| Jinja2 | 3.1.6 | BSD-3-Clause | Base/extras | [source](https://github.com/pallets/jinja/); [Full license and notices](third_party/licenses/jinja2-3-1-6.txt) |
| jiter | 0.17.0 | MIT | Base/extras | [source](https://github.com/pydantic/jiter/); [Full license and notices](third_party/licenses/jiter-0-17-0.txt) |
| joblib | 1.6.0 | BSD-3-Clause | Base/extras | [source](https://github.com/joblib/joblib); [Full license and notices](third_party/licenses/joblib-1-6-0.txt) |
| jsonschema | 4.26.0 | MIT | Base/extras | [source](https://github.com/python-jsonschema/jsonschema); [Full license and notices](third_party/licenses/jsonschema-4-26-0.txt) |
| jsonschema-specifications | 2025.9.1 | MIT | Base/extras | [source](https://github.com/python-jsonschema/jsonschema-specifications); [Full license and notices](third_party/licenses/jsonschema-specifications-2025-9-1.txt) |
| kernels | 0.14.1 | Apache-2.0 | Serving reference | [source](https://pypi.org/project/kernels/0.14.1/); [Full license and notices](third_party/licenses/kernels-0-14-1.txt) |
| kernels-data | 0.16.2 | Apache-2.0 | Serving reference | [source](https://github.com/huggingface/kernels/tree/v0.16.2/kernels-data); [Full license and notices](third_party/licenses/kernels-data-0-16-2.txt) |
| kubernetes_asyncio | 36.1.0 | Apache-2.0 | Base/extras | [source](https://pypi.org/project/kubernetes_asyncio/36.1.0/); [Full license and notices](third_party/licenses/kubernetes-asyncio-36-1-0.txt) |
| lark | 1.3.1 | MIT | Serving reference | [source](https://github.com/lark-parser/lark); [Full license and notices](third_party/licenses/lark-1-3-1.txt) |
| linkify-it-py | 2.2.0 | MIT | Base/extras | [source](https://github.com/tsutsu3/linkify-it-py); [Full license and notices](third_party/licenses/linkify-it-py-2-2-0.txt) |
| llguidance | 1.9.0 | MIT | Serving reference | [source](https://github.com/microsoft/llguidance); [Full license and notices](third_party/licenses/llguidance-1-9-0.txt) |
| loguru | 0.7.3 | MIT | Serving reference | [source](https://github.com/Delgan/loguru); [Full license and notices](third_party/licenses/loguru-0-7-3.txt) |
| lxml | 6.1.3 | BSD-3-Clause | Base/extras | [source](https://github.com/lxml/lxml); [Full license and notices](third_party/licenses/lxml-6-1-3.txt) |
| Mako | 1.4.3 | MIT | Base/extras | [source](https://www.makotemplates.org/); [Full license and notices](third_party/licenses/mako-1-4-3.txt) |
| Markdown | 3.11 | BSD-3-Clause | Base/extras | [source](https://github.com/Python-Markdown/markdown); [Full license and notices](third_party/licenses/markdown-3-11.txt) |
| markdown-it-py | 4.2.0 | MIT | Base/extras | [source](https://github.com/executablebooks/markdown-it-py); [Full license and notices](third_party/licenses/markdown-it-py-4-2-0.txt) |
| MarkupSafe | 3.0.4 | BSD-3-Clause | Base/extras | [source](https://github.com/pallets/markupsafe/); [Full license and notices](third_party/licenses/markupsafe-3-0-4.txt) |
| matplotlib | 3.11.2 | Matplotlib license with third-party notices (including MIT and Apache-2.0) | Base/extras | [source](https://github.com/matplotlib/matplotlib); [Full license and notices](third_party/licenses/matplotlib-3-11-2.txt) |
| mcp | 2.3.0 | MIT | Base/extras | [source](https://github.com/modelcontextprotocol/python-sdk); [Full license and notices](third_party/licenses/mcp-2-3-0.txt) |
| mcp-types | 2.3.0 | MIT | Base/extras | [source](https://github.com/modelcontextprotocol/python-sdk); [Full license and notices](third_party/licenses/mcp-types-2-3-0.txt) |
| mdit-py-plugins | 0.6.1 | MIT | Base/extras | [source](https://github.com/executablebooks/mdit-py-plugins); [Full license and notices](third_party/licenses/mdit-py-plugins-0-6-1.txt) |
| mdurl | 0.1.2 | MIT | Base/extras | [source](https://github.com/executablebooks/mdurl); [Full license and notices](third_party/licenses/mdurl-0-1-2.txt) |
| memray | 1.20.0 | Apache-2.0 | Base/extras | [source](https://github.com/bloomberg/memray); [Full license and notices](third_party/licenses/memray-1-20-0.txt) |
| mistral_common | 1.12.0 | Apache-2.0 | Serving reference | [source](https://pypi.org/project/mistral_common/1.12.0/); [Full license and notices](third_party/licenses/mistral-common-1-12-0.txt) |
| ml_dtypes | 0.6.0 | Apache-2.0 | Base/extras | [source](https://pypi.org/project/ml_dtypes/0.6.0/); [Full license and notices](third_party/licenses/ml-dtypes-0-6-0.txt) |
| mlflow | 3.16.1 | Apache-2.0 | Base/extras | [source](https://pypi.org/project/mlflow/3.16.1/); [Full license and notices](third_party/licenses/mlflow-3-16-1.txt) |
| mlflow-skinny | 3.16.1 | Apache-2.0 | Base/extras | [source](https://pypi.org/project/mlflow-skinny/3.16.1/); [Full license and notices](third_party/licenses/mlflow-skinny-3-16-1.txt) |
| mlflow-tracing | 3.16.1 | Apache-2.0 | Base/extras | [source](https://pypi.org/project/mlflow-tracing/3.16.1/); [Full license and notices](third_party/licenses/mlflow-tracing-3-16-1.txt) |
| modal | 1.6.1 | Apache-2.0 | Base/extras | [source](https://github.com/modal-labs/modal-client); [Full license and notices](third_party/licenses/modal-1-6-1.txt) |
| modelscope | 1.40.1 | Apache-2.0 | Serving reference | [source](https://github.com/modelscope/modelscope); [Full license and notices](third_party/licenses/modelscope-1-40-1.txt) |
| modelscope-hub | 0.4.5 | Apache-2.0 | Serving reference | [source](https://pypi.org/project/modelscope-hub/0.4.5/); [Full license and notices](third_party/licenses/modelscope-hub-0-4-5.txt) |
| mpmath | 1.3.0 | BSD-3-Clause | Base/extras | [source](https://github.com/fredrik-johansson/mpmath); [Full license and notices](third_party/licenses/mpmath-1-3-0.txt) |
| msgpack | 1.2.3 | Apache-2.0 | Base/extras | [source](https://github.com/msgpack/msgpack-python/); [Full license and notices](third_party/licenses/msgpack-1-2-3.txt) |
| multidict | 6.9.1 | Apache-2.0 | Base/extras | [source](https://github.com/aio-libs/multidict); [Full license and notices](third_party/licenses/multidict-6-9-1.txt) |
| multiprocess | 0.70.19 | BSD-3-Clause | Base/extras | [source](https://github.com/uqfoundation/multiprocess); [Full license and notices](third_party/licenses/multiprocess-0-70-19.txt) |
| narwhals | 2.26.0 | MIT | Base/extras | [source](https://github.com/narwhals-dev/narwhals); [Full license and notices](third_party/licenses/narwhals-2-26-0.txt) |
| nccl4py | 0.6.0 | Apache-2.0 | Serving reference | [source](https://github.com/NVIDIA/nccl); [Full license and notices](third_party/licenses/nccl4py-0-6-0.txt) |
| networkx | 3.7 | BSD-3-Clause | Base/extras | [source](https://github.com/networkx/networkx); [Full license and notices](third_party/licenses/networkx-3-7.txt) |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Base/extras | [source](https://pypi.org/project/numpy/2.5.3/); [Full license and notices](third_party/licenses/numpy-2-5-3.txt) |
| nvidia-cudnn-frontend | 1.30.0 | Apache-2.0 AND MIT | Serving reference | [source](https://github.com/NVIDIA/cudnn-frontend); [Full license and notices](third_party/licenses/nvidia-cudnn-frontend-1-30-0.txt) |
| nvidia-ml-py | 13.615.71 | BSD-3-Clause | Base/extras | [source](https://forums.developer.nvidia.com); [Full license and notices](third_party/licenses/nvidia-ml-py-13-615-71.txt) |
| nvidia-nvtx | 13.0.85 | Apache-2.0 | Base/extras | [source](https://developer.nvidia.com/cuda-zone); [Full license and notices](third_party/licenses/nvidia-nvtx-13-0-85.txt) |
| nvidia-resiliency-ext | 0.6.0 | Apache-2.0 | Base/extras | [source](https://github.com/NVIDIA/nvidia-resiliency-ext); [Full license and notices](third_party/licenses/nvidia-resiliency-ext-0-6-0.txt) |
| omegaconf | 2.3.1 | BSD-3-Clause | Base/extras | [source](https://github.com/omry/omegaconf); [Full license and notices](third_party/licenses/omegaconf-2-3-1.txt) |
| onnx | 1.23.1 | Apache-2.0 | Base/extras | [source](https://github.com/onnx/onnx); [Full license and notices](third_party/licenses/onnx-1-23-1.txt) |
| onnx-ir | 1.0.0 | Apache-2.0 | Base/extras | [source](https://github.com/onnx/ir-py); [Full license and notices](third_party/licenses/onnx-ir-1-0-0.txt) |
| onnxscript | 0.7.2 | MIT | Base/extras | [source](https://github.com/microsoft/onnxscript); [Full license and notices](third_party/licenses/onnxscript-0-7-2.txt) |
| openai | 3.24.0 | Apache-2.0 | Base/extras | [source](https://github.com/openai/openai-python); [Full license and notices](third_party/licenses/openai-3-24-0.txt) |
| openai-harmony | 0.0.4 | Apache-2.0 | Serving reference | [source](https://pypi.org/project/openai-harmony/0.0.4/); [Full license and notices](third_party/licenses/openai-harmony-0-0-4.txt) |
| opencensus | 0.11.4 | Apache-2.0 | Base/extras | [source](https://github.com/census-instrumentation/opencensus-python); [Full license and notices](third_party/licenses/opencensus-0-11-4.txt) |
| opencensus-context | 0.1.3 | Apache-2.0 | Base/extras | [source](https://github.com/census-instrumentation/opencensus-python/tree/master/context/opencensus-context); [Full license and notices](third_party/licenses/opencensus-context-0-1-3.txt) |
| opentelemetry-api | 1.45.0 | Apache-2.0 | Base/extras | [source](https://github.com/open-telemetry/opentelemetry-python); [Full license and notices](third_party/licenses/opentelemetry-api-1-45-0.txt) |
| opentelemetry-exporter-http-transport | 0.66b0 | Apache-2.0 | Base/extras | [source](https://github.com/open-telemetry/opentelemetry-python); [Full license and notices](third_party/licenses/opentelemetry-exporter-http-transport-0-66b0.txt) |
| opentelemetry-exporter-otlp-common | 0.66b0 | Apache-2.0 | Base/extras | [source](https://github.com/open-telemetry/opentelemetry-python); [Full license and notices](third_party/licenses/opentelemetry-exporter-otlp-common-0-66b0.txt) |
| opentelemetry-exporter-otlp-proto-common | 1.45.0 | Apache-2.0 | Base/extras | [source](https://github.com/open-telemetry/opentelemetry-python); [Full license and notices](third_party/licenses/opentelemetry-exporter-otlp-proto-common-1-45-0.txt) |
| opentelemetry-exporter-otlp-proto-http | 1.45.0 | Apache-2.0 | Base/extras | [source](https://github.com/open-telemetry/opentelemetry-python); [Full license and notices](third_party/licenses/opentelemetry-exporter-otlp-proto-http-1-45-0.txt) |
| opentelemetry-exporter-prometheus | 0.66b0 | Apache-2.0 | Base/extras | [source](https://github.com/open-telemetry/opentelemetry-python); [Full license and notices](third_party/licenses/opentelemetry-exporter-prometheus-0-66b0.txt) |
| opentelemetry-proto | 1.45.0 | Apache-2.0 | Base/extras | [source](https://github.com/open-telemetry/opentelemetry-python); [Full license and notices](third_party/licenses/opentelemetry-proto-1-45-0.txt) |
| opentelemetry-sdk | 1.45.0 | Apache-2.0 | Base/extras | [source](https://github.com/open-telemetry/opentelemetry-python); [Full license and notices](third_party/licenses/opentelemetry-sdk-1-45-0.txt) |
| opentelemetry-semantic-conventions | 0.66b0 | Apache-2.0 | Base/extras | [source](https://github.com/open-telemetry/opentelemetry-python); [Full license and notices](third_party/licenses/opentelemetry-semantic-conventions-0-66b0.txt) |
| orjson | 3.12.0 | MPL-2.0 AND (Apache-2.0 OR MIT) | Base/extras | [source](https://pypi.org/project/orjson/3.12.0/); [Full license and notices](third_party/licenses/orjson-3-12-0.txt) |
| outlines | 0.1.11 | Apache-2.0 | Serving reference | [source](https://pypi.org/project/outlines/0.1.11/); [Full license and notices](third_party/licenses/outlines-0-1-11.txt) |
| outlines_core | 0.1.26 | Apache-2.0 | Serving reference | [source](https://pypi.org/project/outlines_core/0.1.26/); [Full license and notices](third_party/licenses/outlines-core-0-1-26.txt) |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | Base/extras | [source](https://github.com/pypa/packaging); [Full license and notices](third_party/licenses/packaging-26-3.txt) |
| pandas | 3.0.6 | BSD-3-Clause | Base/extras | [source](https://pypi.org/project/pandas/3.0.6/); [Full license and notices](third_party/licenses/pandas-3-0-6.txt) |
| partial-json-parser | 0.2.1.1.post7 | MIT | Serving reference | [source](https://pypi.org/project/partial-json-parser/0.2.1.1.post7/); [Full license and notices](third_party/licenses/partial-json-parser-0-2-1-1-post7.txt) |
| pillow | 12.3.0 | MIT-CMU | Base/extras | [source](https://github.com/python-pillow/Pillow); [Full license and notices](third_party/licenses/pillow-12-3-0.txt) |
| platformdirs | 4.12.3 | MIT | Base/extras | [source](https://github.com/tox-dev/platformdirs); [Full license and notices](third_party/licenses/platformdirs-4-12-3.txt) |
| pluggy | 1.6.0 | MIT | Base/extras | [source](https://pypi.org/project/pluggy/1.6.0/); [Full license and notices](third_party/licenses/pluggy-1-6-0.txt) |
| polars | 1.42.1 | MIT | Base/extras | [source](https://github.com/pola-rs/polars); [Full license and notices](third_party/licenses/polars-1-42-1.txt) |
| polars-runtime-32 | 1.42.1 | MIT | Base/extras | [source](https://github.com/pola-rs/polars); [Full license and notices](third_party/licenses/polars-runtime-32-1-42-1.txt) |
| prettytable | 3.18.0 | BSD-3-Clause | Base/extras | [source](https://github.com/prettytable/prettytable); [Full license and notices](third_party/licenses/prettytable-3-18-0.txt) |
| prometheus_client | 0.26.0 | Apache-2.0 AND BSD-2-Clause | Base/extras | [source](https://github.com/prometheus/client_python); [Full license and notices](third_party/licenses/prometheus-client-0-26-0.txt) |
| propcache | 0.5.4 | Apache-2.0 | Base/extras | [source](https://github.com/aio-libs/propcache); [Full license and notices](third_party/licenses/propcache-0-5-4.txt) |
| proto-plus | 1.29.0 | Apache-2.0 | Base/extras | [source](https://github.com/googleapis/google-cloud-python); [Full license and notices](third_party/licenses/proto-plus-1-29-0.txt) |
| protobuf | 7.36.2 | BSD-3-Clause | Base/extras | [source](https://developers.google.com/protocol-buffers/); [Full license and notices](third_party/licenses/protobuf-7-36-2.txt) |
| protobuf-py | 0.1.1 | Apache-2.0 | Base/extras | [source](https://github.com/bufbuild/protobuf-py); [Full license and notices](third_party/licenses/protobuf-py-0-1-1.txt) |
| protobuf-py-ext | 0.1.1 | Apache-2.0 | Base/extras | [source](https://github.com/bufbuild/protobuf-py); [Full license and notices](third_party/licenses/protobuf-py-ext-0-1-1.txt) |
| psutil | 7.2.2 | BSD-3-Clause | Base/extras | [source](https://github.com/giampaolo/psutil); [Full license and notices](third_party/licenses/psutil-7-2-2.txt) |
| py-spy | 0.4.2 | MIT | Base/extras | [source](https://github.com/benfred/py-spy); [Full license and notices](third_party/licenses/py-spy-0-4-2.txt) |
| pyarrow | 25.0.1 | Apache-2.0 | Base/extras | [source](https://github.com/apache/arrow); [Full license and notices](third_party/licenses/pyarrow-25-0-1.txt) |
| pyasn1 | 0.6.4 | BSD-2-Clause | Base/extras | [source](https://github.com/pyasn1/pyasn1); [Full license and notices](third_party/licenses/pyasn1-0-6-4.txt) |
| pyasn1_modules | 0.4.2 | BSD-2-Clause | Base/extras | [source](https://github.com/pyasn1/pyasn1-modules); [Full license and notices](third_party/licenses/pyasn1-modules-0-4-2.txt) |
| pybase64 | 1.5.0 | BSD-2-Clause | Base/extras | [source](https://github.com/mayeut/pybase64); [Full license and notices](third_party/licenses/pybase64-1-5-0.txt) |
| pycparser | 3.0 | BSD-3-Clause | Base/extras | [source](https://github.com/eliben/pycparser); [Full license and notices](third_party/licenses/pycparser-3-0.txt) |
| pycryptodomex | 3.23.0 | BSD-2-Clause and public-domain portions | Base/extras | [source](https://github.com/Legrandin/pycryptodome/); [Full license and notices](third_party/licenses/pycryptodomex-3-23-0.txt) |
| pydantic | 2.13.5 | MIT | Base/extras | [source](https://github.com/pydantic/pydantic); [Full license and notices](third_party/licenses/pydantic-2-13-5.txt) |
| pydantic-extra-types | 2.11.1 | MIT | Serving reference | [source](https://github.com/pydantic/pydantic-extra-types); [Full license and notices](third_party/licenses/pydantic-extra-types-2-11-1.txt) |
| pydantic_core | 2.46.5 | MIT | Base/extras | [source](https://github.com/pydantic/pydantic/tree/main/pydantic-core); [Full license and notices](third_party/licenses/pydantic-core-2-46-5.txt) |
| Pygments | 2.21.0 | BSD-2-Clause | Base/extras | [source](https://github.com/pygments/pygments); [Full license and notices](third_party/licenses/pygments-2-21-0.txt) |
| PyJWT | 2.15.1 | MIT | Base/extras | [source](https://github.com/jpadilla/pyjwt); [Full license and notices](third_party/licenses/pyjwt-2-15-1.txt) |
| pylatexenc | 2.11 | MIT | Base/extras | [source](https://github.com/phfaist/pylatexenc); [Full license and notices](third_party/licenses/pylatexenc-2-11.txt) |
| pyparsing | 3.3.3 | MIT | Base/extras | [source](https://github.com/pyparsing/pyparsing.git); [Full license and notices](third_party/licenses/pyparsing-3-3-3.txt) |
| pyqwest | 0.10.0 | MIT | Base/extras | [source](https://github.com/curioswitch/pyqwest.git); [Full license and notices](third_party/licenses/pyqwest-0-10-0.txt) |
| pytest | 9.1.1 | MIT | Base/extras | [source](https://github.com/pytest-dev/pytest); [Full license and notices](third_party/licenses/pytest-9-1-1.txt) |
| pytest-asyncio | 1.4.0 | Apache-2.0 | Base/extras | [source](https://github.com/pytest-dev/pytest-asyncio); [Full license and notices](third_party/licenses/pytest-asyncio-1-4-0.txt) |
| python-dateutil | 2.9.0.post0 | Apache-2.0 OR BSD-3-Clause | Base/extras | [source](https://github.com/dateutil/dateutil); [Full license and notices](third_party/licenses/python-dateutil-2-9-0-post0.txt) |
| python-discovery | 1.6.1 | MIT | Base/extras | [source](https://github.com/tox-dev/python-discovery); [Full license and notices](third_party/licenses/python-discovery-1-6-1.txt) |
| python-dotenv | 1.2.4 | BSD-3-Clause | Base/extras | [source](https://github.com/theskumar/python-dotenv); [Full license and notices](third_party/licenses/python-dotenv-1-2-4.txt) |
| python-multipart | 0.0.32 | Apache-2.0 | Base/extras | [source](https://github.com/Kludex/python-multipart); [Full license and notices](third_party/licenses/python-multipart-0-0-32.txt) |
| PyYAML | 6.0.3 | MIT | Base/extras | [source](https://github.com/yaml/pyyaml); [Full license and notices](third_party/licenses/pyyaml-6-0-3.txt) |
| quack-kernels | 0.6.4 | Apache-2.0 | Serving reference | [source](https://pypi.org/project/quack-kernels/0.6.4/); [Full license and notices](third_party/licenses/quack-kernels-0-6-4.txt) |
| qwen-vl-utils | 0.0.14 | Apache-2.0 | Base/extras | [source](https://github.com/QwenLM/Qwen2-VL.git); [Full license and notices](third_party/licenses/qwen-vl-utils-0-0-14.txt) |
| ray | 2.59.0 | Apache-2.0 | Base/extras | [source](https://github.com/ray-project/ray); [Full license and notices](third_party/licenses/ray-2-59-0.txt) |
| referencing | 0.37.0 | MIT | Base/extras | [source](https://github.com/python-jsonschema/referencing); [Full license and notices](third_party/licenses/referencing-0-37-0.txt) |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python | Base/extras | [source](https://github.com/mrabarnett/mrab-regex); [Full license and notices](third_party/licenses/regex-2026-9-29.txt) |
| requests | 2.34.2 | Apache-2.0 | Base/extras | [source](https://github.com/psf/requests); [Full license and notices](third_party/licenses/requests-2-34-2.txt) |
| rich | 15.0.0 | MIT | Base/extras | [source](https://github.com/Textualize/rich); [Full license and notices](third_party/licenses/rich-15-0-0.txt) |
| ring-flash-attn | 0.1.8 | MIT | Base/extras | [source](https://github.com/zhuzilin/ring-flash-attention); [Full license and notices](third_party/licenses/ring-flash-attn-0-1-8.txt) |
| rpds-py | 2026.6.3 | MIT | Base/extras | [source](https://github.com/crate-py/rpds); [Full license and notices](third_party/licenses/rpds-py-2026-6-3.txt) |
| safetensors | 0.8.0 | Apache-2.0 | Base/extras | [source](https://github.com/huggingface/safetensors); [Full license and notices](third_party/licenses/safetensors-0-8-0.txt) |
| scikit-learn | 1.9.1 | BSD-3-Clause | Base/extras | [source](https://pypi.org/project/scikit-learn/1.9.1/); [Full license and notices](third_party/licenses/scikit-learn-1-9-1.txt) |
| setproctitle | 1.3.8 | BSD-3-Clause | Base/extras | [source](https://github.com/dvarrazzo/py-setproctitle); [Full license and notices](third_party/licenses/setproctitle-1-3-8.txt) |
| setuptools | 84.0.0 | MIT | Base/extras | [source](https://github.com/pypa/setuptools); [Full license and notices](third_party/licenses/setuptools-84-0-0.txt) |
| sgl-deep-ep | 0.1.0 | Apache-2.0 | Serving reference | [source](https://pypi.org/project/sgl-deep-ep/0.1.0/); [Full license and notices](third_party/licenses/sgl-deep-ep-0-1-0.txt) |
| sgl-deep-gemm | 0.1.5.post3 | Apache-2.0 | Serving reference | [source](https://github.com/sgl-project/DeepGEMM/tree/release); [Full license and notices](third_party/licenses/sgl-deep-gemm-0-1-5-post3.txt) |
| sglang-kernel | 0.4.6.post1 | Apache-2.0 | Serving reference | [source](https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/aot); [Full license and notices](third_party/licenses/sglang-kernel-0-4-6-post1.txt) |
| sglang-router | 0.3.2 | Apache-2.0 | Base/extras | [source](https://pypi.org/project/sglang-router/0.3.2/); [Full license and notices](third_party/licenses/sglang-router-0-3-2.txt) |
| shellingham | 1.5.4 | ISC | Base/extras | [source](https://github.com/sarugaku/shellingham); [Full license and notices](third_party/licenses/shellingham-1-5-4.txt) |
| six | 1.17.0 | MIT | Base/extras | [source](https://github.com/benjaminp/six); [Full license and notices](third_party/licenses/six-1-17-0.txt) |
| skops | 0.16.0 | MIT | Base/extras | [source](http://github.com/skops-dev/skops); [Full license and notices](third_party/licenses/skops-0-16-0.txt) |
| smart_open | 8.0.2 | MIT | Base/extras | [source](https://github.com/piskvorky/smart_open); [Full license and notices](third_party/licenses/smart-open-8-0-2.txt) |
| smg-grpc-proto | 0.4.22 | Apache-2.0 | Serving reference | [source](https://github.com/smg-project/smg); [Full license and notices](third_party/licenses/smg-grpc-proto-0-4-22.txt) |
| smg-grpc-servicer | 0.13.0 | Apache-2.0 | Serving reference | [source](https://github.com/smg-project/smg); [Full license and notices](third_party/licenses/smg-grpc-servicer-0-13-0.txt) |
| smmap | 5.0.3 | BSD-3-Clause | Base/extras | [source](https://github.com/gitpython-developers/smmap); [Full license and notices](third_party/licenses/smmap-5-0-3.txt) |
| sniffio | 1.3.1 | MIT OR Apache-2.0 | Base/extras | [source](https://github.com/python-trio/sniffio); [Full license and notices](third_party/licenses/sniffio-1-3-1.txt) |
| sortedcontainers | 2.4.0 | Apache-2.0 | Base/extras | [source](http://www.grantjenks.com/docs/sortedcontainers/); [Full license and notices](third_party/licenses/sortedcontainers-2-4-0.txt) |
| SQLAlchemy | 2.1.3 | MIT | Base/extras | [source](https://github.com/sqlalchemy/sqlalchemy); [Full license and notices](third_party/licenses/sqlalchemy-2-1-3.txt) |
| sqlparse | 0.6.0 | BSD-3-Clause | Base/extras | [source](https://github.com/andialbrecht/sqlparse); [Full license and notices](third_party/licenses/sqlparse-0-6-0.txt) |
| sse-starlette | 3.5.0 | BSD-3-Clause | Base/extras | [source](https://github.com/sysid/sse-starlette); [Full license and notices](third_party/licenses/sse-starlette-3-5-0.txt) |
| starlette | 1.7.0 | BSD-3-Clause | Base/extras | [source](https://github.com/Kludex/starlette); [Full license and notices](third_party/licenses/starlette-1-7-0.txt) |
| syllapy | 0.8.0 | MIT | Optional integration | [source](https://github.com/mholtzscher/syllapy); [Full license and notices](third_party/licenses/syllapy-0-8-0.txt) |
| sympy | 1.14.0 | BSD-3-Clause | Base/extras | [source](https://github.com/sympy/sympy); [Full license and notices](third_party/licenses/sympy-1-14-0.txt) |
| synchronicity | 0.12.6 | Apache-2.0 | Base/extras | [source](https://pypi.org/project/synchronicity/0.12.6/); [Full license and notices](third_party/licenses/synchronicity-0-12-6.txt) |
| tensorboard | 2.21.0 | Apache-2.0 | Base/extras | [source](https://github.com/tensorflow/tensorboard); [Full license and notices](third_party/licenses/tensorboard-2-21-0.txt) |
| tensorboard-data-server | 0.7.2 | Apache-2.0 | Base/extras | [source](https://github.com/tensorflow/tensorboard/tree/master/tensorboard/data/server); [Full license and notices](third_party/licenses/tensorboard-data-server-0-7-2.txt) |
| textual | 8.2.8 | MIT | Base/extras | [source](https://github.com/Textualize/textual); [Full license and notices](third_party/licenses/textual-8-2-8.txt) |
| threadpoolctl | 3.7.0 | BSD-3-Clause | Base/extras | [source](https://pypi.org/project/threadpoolctl/3.7.0/); [Full license and notices](third_party/licenses/threadpoolctl-3-7-0.txt) |
| tilelang | 0.1.11 | MIT | Serving reference | [source](https://pypi.org/project/tilelang/0.1.11/); [Full license and notices](third_party/licenses/tilelang-0-1-11.txt) |
| tinker | 0.26.2 | Apache-2.0 | Optional integration | [source](https://github.com/thinking-machines-lab/tinker); [Full license and notices](third_party/licenses/tinker-0-26-2.txt) |
| tokenizers | 0.22.2 | Apache-2.0 | Base/extras | [source](https://github.com/huggingface/tokenizers); [Full license and notices](third_party/licenses/tokenizers-0-22-2.txt) |
| tokenspeed-mla | 0.1.8 | MIT | Serving reference | [source](https://github.com/lightseekorg/tokenspeed); [Full license and notices](third_party/licenses/tokenspeed-mla-0-1-8.txt) |
| tokenspeed-triton | 3.8.10.post20260920 | MIT | Serving reference | [source](https://github.com/triton-lang/triton/); [Full license and notices](third_party/licenses/tokenspeed-triton-3-8-10-post20260920.txt) |
| toml | 0.10.2 | MIT | Base/extras | [source](https://github.com/uiri/toml); [Full license and notices](third_party/licenses/toml-0-10-2.txt) |
| torch | 2.14.1 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | Base/extras | [source](https://github.com/pytorch/pytorch); [Full license and notices](third_party/licenses/torch-2-14-1.txt) |
| torch_c_dlpack_ext | 0.1.5 | Apache-2.0 | Serving reference | [source](https://pypi.org/project/torch_c_dlpack_ext/0.1.5/); [Full license and notices](third_party/licenses/torch-c-dlpack-ext-0-1-5.txt) |
| torch_memory_saver | 0.0.10 | MIT | Serving reference | [source](https://pypi.org/project/torch_memory_saver/0.0.10/); [Full license and notices](third_party/licenses/torch-memory-saver-0-0-10.txt) |
| torchcodec | 0.15.0+cu130 | BSD-3-Clause | Serving reference | [source](https://pypi.org/project/torchcodec/0.15.0+cu130/); [Full license and notices](third_party/licenses/torchcodec-0-15-0-cu130.txt) |
| torchft-nightly | 2026.4.3 | BSD-3-Clause | Base/extras | [source](https://github.com/pytorch/torchft); [Full license and notices](third_party/licenses/torchft-nightly-2026-4-3.txt) |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | Base/extras | [source](https://pypi.org/project/tqdm/4.70.1/); [Full license and notices](third_party/licenses/tqdm-4-70-1.txt) |
| transformers | 5.12.1 | Apache-2.0 | Base/extras | [source](https://github.com/huggingface/transformers); [Full license and notices](third_party/licenses/transformers-5-12-1.txt) |
| triton | 3.8.0 | MIT | Base/extras | [source](https://github.com/triton-lang/triton/); [Full license and notices](third_party/licenses/triton-3-8-0.txt) |
| truststore | 0.10.4 | MIT | Base/extras | [source](https://github.com/sethmlarson/truststore); [Full license and notices](third_party/licenses/truststore-0-10-4.txt) |
| typer | 0.27.2 | MIT | Base/extras | [source](https://github.com/fastapi/typer); [Full license and notices](third_party/licenses/typer-0-27-2.txt) |
| types-certifi | 2021.10.8.3 | Apache-2.0 | Base/extras | [source](https://github.com/python/typeshed); [Full license and notices](third_party/licenses/types-certifi-2021-10-8-3.txt) |
| types-toml | 0.10.8.20260518 | Apache-2.0 | Base/extras | [source](https://github.com/python/typeshed); [Full license and notices](third_party/licenses/types-toml-0-10-8-20260518.txt) |
| typing-inspection | 0.4.4 | MIT | Base/extras | [source](https://github.com/pydantic/typing-inspection); [Full license and notices](third_party/licenses/typing-inspection-0-4-4.txt) |
| urllib3 | 2.8.0 | MIT | Base/extras | [source](https://pypi.org/project/urllib3/2.8.0/); [Full license and notices](third_party/licenses/urllib3-2-8-0.txt) |
| uvicorn | 0.54.0 | BSD-3-Clause | Base/extras | [source](https://github.com/Kludex/uvicorn); [Full license and notices](third_party/licenses/uvicorn-0-54-0.txt) |
| uvloop | 0.22.1 | MIT | Serving reference | [source](https://pypi.org/project/uvloop/0.22.1/); [Full license and notices](third_party/licenses/uvloop-0-22-1.txt) |
| virtualenv | 21.14.5 | MIT | Base/extras | [source](https://github.com/pypa/virtualenv); [Full license and notices](third_party/licenses/virtualenv-21-14-5.txt) |
| wandb | 0.30.0 | MIT | Base/extras | [source](https://github.com/wandb/wandb); [Full license and notices](third_party/licenses/wandb-0-30-0.txt) |
| watchfiles | 1.3.0 | MIT | Base/extras | [source](https://github.com/samuelcolvin/watchfiles); [Full license and notices](third_party/licenses/watchfiles-1-3-0.txt) |
| wcmatch | 11.0.1 | MIT | Base/extras | [source](https://github.com/facelessuser/wcmatch); [Full license and notices](third_party/licenses/wcmatch-11-0-1.txt) |
| wcwidth | 0.9.1 | MIT | Base/extras | [source](https://github.com/jquast/wcwidth); [Full license and notices](third_party/licenses/wcwidth-0-9-1.txt) |
| Werkzeug | 3.1.9 | BSD-3-Clause | Base/extras | [source](https://github.com/pallets/werkzeug/); [Full license and notices](third_party/licenses/werkzeug-3-1-9.txt) |
| wrapt | 2.5.0 | BSD-2-Clause | Base/extras | [source](https://github.com/GrahamDumpleton/wrapt); [Full license and notices](third_party/licenses/wrapt-2-5-0.txt) |
| xgrammar | 0.2.1 | Apache-2.0 | Serving reference | [source](https://xgrammar.mlc.ai/); [Full license and notices](third_party/licenses/xgrammar-0-2-1.txt) |
| xxhash | 3.7.1 | BSD-2-Clause | Base/extras | [source](https://github.com/ifduyue/python-xxhash); [Full license and notices](third_party/licenses/xxhash-3-7-1.txt) |
| yarl | 1.25.1 | Apache-2.0 | Base/extras | [source](https://github.com/aio-libs/yarl); [Full license and notices](third_party/licenses/yarl-1-25-1.txt) |
| zipp | 4.1.1 | MIT | Base/extras | [source](https://github.com/jaraco/zipp); [Full license and notices](third_party/licenses/zipp-4-1-1.txt) |
| zstandard | 0.25.0 | BSD-3-Clause | Base/extras | [source](https://github.com/indygreg/python-zstandard); [Full license and notices](third_party/licenses/zstandard-0-25-0.txt) |

## Maintenance and provenance notes

When adding or changing dependencies, update the relevant entry and preserve the copyright, LICENSE, and NOTICE material from the version actually used. The [machine-readable inventory](third_party/inventory.json) records source links, license evidence revisions, and SHA-256 checksums for every local notice file.

The ProRL-Agent-Server streaming adaptation, DeepSeek-V3.2 encoder, FlashInfer-adapted fake-QAT kernel, DeepSeek-V4 conversion helper, MindSpeed source patch, and TileKernels/TileLang source patches are absent from this release. External FlashInfer and TileLang dependencies are still listed where applicable.

The Typer issue-comment attribution above has no explicit license statement in the credited comment. Its credit is preserved, but the repository MIT license should not be treated as confirmation of the comment’s separate licensing. Several inherited adaptations also lack recorded copied-source revisions; their entries distinguish that from the license evidence that was available.
