# Dependency review: miles-olmo-core

Updated 2026-10-05 for the source-only release at `664f4a8c346828e8ee1d996859e2d373ce54d7f7`. This list replaces the former Miles review scope. It separates **included upstream/adapted source** from **user-installed dependencies — not distributed**. It also reconciles every row of the previous submission.

User-installed packages, their native/Rust components, container images, model weights, and datasets are not bundled in this source release. Installation commands in a recipe do not include the installed package in the repository. Included adaptations and patch context are listed separately, including NVIDIA-origin code that retains its original notices.

Package versions below are inspected references from the 2026-10-04 Python 3.12 base/extras resolution, supplemented by optional-serving review evidence and inherited build recipes. This is not a new dependency lock or a complete native-component SBOM. A reference does not imply that an optional backend is supported by this OLMo-focused fork.

Full attribution for included material is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The [machine-readable review](third_party/dependency-review.json) contains all rows, dependency scopes, and prior-submission dispositions.

## Changes from the previous review

- Removed the ProRL-Agent-Server streaming adaptation, DeepSeek-V3.2 encoder, FlashInfer-adapted fake-QAT kernel, and DeepSeek-V4 conversion helper. External FlashInfer remains a user-installed serving dependency.
- Removed the MindSpeed library patch. The inherited NPU guide still references a separately installed MindSpeed version; the library is not distributed.
- Removed TileKernels/TileLang source patches. The optional CUDA recipe still selects external TileLang 0.1.14; its historical collaboration note has not been declared resolved.
- Removed the examples and their dedicated dependency installation instructions; the prior-submission reconciliation identifies those entries.
- Restricted repository attribution notices to included source and patch context. User-installed reference packages are labeled not distributed, including NVIDIA/vendor-proprietary, LGPL, MPL, and other license families.

## Included upstream, adapted source, and patch context

A reference or license evidence revision is not automatically a copied-source revision. The notices retain the known provenance gaps, including the Typer issue-comment license and mbridge adapter provenance. No license clearance is asserted by this list.

| Name | Reference | License | URL | Distribution |
| --- | --- | --- | --- | --- |
| deepscaler | e6080ccd974eb64bd3430f0b36108244a6fee330 | MIT | https://github.com/agentica-project/deepscaler | Included upstream/adapted source or retained patch context |
| flash-attention-source | main | BSD-3-Clause | https://github.com/Dao-AILab/flash-attention | Included upstream/adapted source or retained patch context |
| lm-evaluation-harness | main | MIT (upstream); Apache-2.0 notices retained in Miles adaptation | https://github.com/EleutherAI/lm-evaluation-harness | Included upstream/adapted source or retained patch context |
| mbridge | 89eb10887887bc74853f89a4de258c0702932a1c | BSD-3-Clause AND Apache-2.0 AND MIT | https://github.com/ISEEKYAN/mbridge | Included upstream/adapted source or retained patch context |
| megatron-bridge | 2e09c234a3272285140224d0d698593418b55ba5 | Apache-2.0 | https://github.com/radixark/Megatron-Bridge | Included upstream/adapted source or retained patch context |
| megatron-lm-fork | 4716f75475c78e2fc2c6f0d3af095f1681b770b4 | BSD-3-Clause; Apache-2.0 and MIT portions | https://github.com/radixark/Megatron-LM | Included upstream/adapted source or retained patch context |
| megatron-lm-source | b1efb3c7126ef7615e8c333432d76e08038e17ff | BSD-3-Clause; Apache-2.0 and MIT portions | https://github.com/NVIDIA/Megatron-LM | Included upstream/adapted source or retained patch context |
| miles-upstream | dbbab1566ae438f7202fff653eae938e07b1d4b6 | Apache-2.0 | https://github.com/radixark/miles | Included upstream/adapted source or retained patch context |
| openrlhf | 10c733694ed9fbb78a0a2ff6a05efc7401584d46 | Apache-2.0 | https://github.com/OpenRLHF/OpenRLHF | Included upstream/adapted source or retained patch context |
| pai-megatron-patch | 2b201af08336dea0403df7c6b497c964cf5a2e75 | Apache-2.0 | https://github.com/alibaba/Pai-Megatron-Patch | Included upstream/adapted source or retained patch context |
| pytorch | v2.11.0 | BSD-3-Clause | https://github.com/pytorch/pytorch | Included upstream/adapted source or retained patch context |
| ray-source | 161849364a784442cc659fb9780f1a6adee85fce | Apache-2.0 | https://github.com/ray-project/ray | Included upstream/adapted source or retained patch context |
| sglang-source | 3145136dcd1238754e0ea2b2ffd546532119c71c | Apache-2.0 | https://github.com/sgl-project/sglang | Included upstream/adapted source or retained patch context |
| slime | main | Apache-2.0 | https://github.com/THUDM/slime | Included upstream/adapted source or retained patch context |
| torchft-source | main | BSD-3-Clause | https://github.com/pytorch/torchft | Included upstream/adapted source or retained patch context |
| transformer-engine | 2e559f062497bef768dfbe9d7e45548fadeca80a | Apache-2.0 | https://github.com/NVIDIA/TransformerEngine | Included upstream/adapted source or retained patch context |
| transformers-source | 38a08b6e8ae35857109cedad75377997fecbf9d0 | Apache-2.0 | https://github.com/huggingface/transformers | Included upstream/adapted source or retained patch context |
| triton-source | main | MIT | https://github.com/triton-lang/triton | Included upstream/adapted source or retained patch context |
| typer-source | master | MIT (upstream Typer); credited issue-comment license not established | https://github.com/fastapi/typer | Included upstream/adapted source or retained patch context |
| verl | 468adf22c43b744348051fccd7a5d830c6c3c36a | Apache-2.0 | https://github.com/verl-project/verl | Included upstream/adapted source or retained patch context |

## User-installed dependency reference — not distributed

Every entry below is **not distributed** with this source release. Full packages remain external even where a separately identified adaptation or patch context is included above.

| Name | Inspected version or recipe reference | License | URL | Reference scope | Distribution |
| --- | --- | --- | --- | --- | --- |
| absl-py | 2.5.0 | Apache-2.0 | https://github.com/abseil/abseil-py | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| accelerate | 1.15.0 | Apache-2.0 | https://github.com/huggingface/accelerate | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| ai2-olmo-core | 3.0.0 | Apache-2.0 | https://github.com/allenai/Olmo-core | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| aiohappyeyeballs | 2.7.1 | PSF-2.0 | https://github.com/aio-libs/aiohappyeyeballs | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| aiohttp | 3.14.3 | Apache-2.0 AND MIT | https://github.com/aio-libs/aiohttp | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| aiohttp-cors | 0.8.1 | Apache-2.0 | https://github.com/aio-libs/aiohttp-cors | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| aiosignal | 1.4.0 | Apache-2.0 | https://github.com/aio-libs/aiosignal | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| airportsdata | 20260905 | MIT | https://github.com/mborsetti/airportsdata/ | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| alembic | 1.20.0 | MIT | https://github.com/sqlalchemy/alembic/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| annotated-doc | 0.0.5 | MIT | https://github.com/fastapi/annotated-doc | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| annotated-types | 0.8.0 | MIT | https://github.com/annotated-types/annotated-types | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| antlr4-python3-runtime | 4.9.3 | BSD-3-Clause | http://www.antlr.org | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| anyio | 4.15.1 | MIT | https://pypi.org/project/anyio/4.15.1/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| apache-tvm-ffi | 0.1.11 | Apache-2.0 | https://github.com/apache/tvm-ffi | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| apex | master | BSD-3-Clause | https://github.com/NVIDIA/apex | Inherited recipe/source integration | Not distributed — user-installed dependency |
| Ascend HDK | 25.3.RC1 | Vendor terms: Legal review required; end-user installed | https://www.hiascend.com/hardware/firmware-drivers/commercial | Inherited optional NPU guide; not the OLMo-core training path | Not distributed — user-installed dependency |
| attrs | 26.1.0 | MIT | https://pypi.org/project/attrs/26.1.0/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| av | 19.0.1 | BSD-3-Clause | https://github.com/PyAV-Org/PyAV | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| blake3 | 1.0.9 | CC0-1.0 OR Apache-2.0 | https://github.com/oconnor663/blake3-py | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| blinker | 1.9.0 | MIT | https://github.com/pallets-eco/blinker/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| blobfile | 3.3.0 | Public Domain | https://github.com/blobfile/blobfile | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| bracex | 3.0.1 | MIT | https://github.com/facelessuser/bracex | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| cachetools | 7.2.0 | MIT | https://github.com/tkem/cachetools/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| CANN Toolkit, kernels, NNAL | 8.5.0 | CANN vendor agreement; component-specific terms require confirmation | https://www.hiascend.com/developer/download | Inherited optional NPU guide; not the OLMo-core training path | Not distributed — user-installed dependency |
| causal-conv1d | v1.6.1 | BSD-3-Clause | https://github.com/Dao-AILab/causal-conv1d | Inherited recipe/source integration | Not distributed — user-installed dependency |
| cbor2 | 6.1.5 | MIT | https://github.com/agronholm/cbor2 | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| certifi | 2026.7.22 | MPL-2.0 | https://github.com/certifi/python-certifi | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| cffi | 2.1.1 | MIT-0 | https://github.com/python-cffi/cffi | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| charset-normalizer | 3.5.2 | MIT | https://pypi.org/project/charset-normalizer/3.5.2/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| chz | 0.4.0 | MIT | https://github.com/openai/chz | External dependency: retained optional Tinker/IFBench integration | Not distributed — user-installed dependency |
| click | 8.5.0 | BSD-3-Clause | https://github.com/pallets/click/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| cloudpickle | 3.1.2 | BSD-3-Clause | https://github.com/cloudpipe/cloudpickle | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| colorful | 0.5.8 | MIT | http://github.com/timofurrer/colorful | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| compressed-tensors | 0.19.0 | Apache-2.0 | https://github.com/vllm-project/compressed-tensors | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| connectrpc | 0.11.1 | Apache-2.0 | https://github.com/connectrpc/connect-py | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| contourpy | 1.4.0 | BSD-3-Clause | https://github.com/contourpy/contourpy | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| cryptography | 50.0.2 | Apache-2.0 OR BSD-3-Clause | https://pypi.org/project/cryptography/50.0.2/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| cuda-bindings | 13.4.3 | Apache-2.0 | https://github.com/NVIDIA/cuda-python | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| cuda-core | 1.2.1 | Apache-2.0 | https://github.com/NVIDIA/cuda-python/tree/main/cuda_core | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| cuda-pathfinder | 1.8.3 | Apache-2.0 | https://github.com/NVIDIA/cuda-python | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| cuda-python | 13.4.1 | Apache-2.0 | https://github.com/NVIDIA/cuda-python/ | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| cuda-tile | 1.6.0rc5 | Apache-2.0 | https://github.com/nvidia/cutile-python | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| cuda-toolkit | 13.0.3.0 | Metadata-only package; component-specific NVIDIA terms apply | https://developer.nvidia.com/cuda-toolkit | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| cycler | 0.12.1 | BSD-3-Clause | https://github.com/matplotlib/cycler | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| databricks-sdk | 0.67.0 | Apache-2.0 | https://pypi.org/project/databricks-sdk/0.67.0/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| dataclass-extensions | 0.5.0 | Apache-2.0 | https://github.com/epwalsh/dataclass-extensions | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| datasets | 5.0.1 | Apache-2.0 | https://github.com/huggingface/datasets | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| defusedxml | 0.7.1 | PSFL | https://github.com/tiran/defusedxml | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| dill | 0.4.1 | BSD-3-Clause | https://github.com/uqfoundation/dill | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| distlib | 0.4.3 | PSF-2.0 | https://github.com/pypa/distlib | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| docker | 7.2.0 | Apache-2.0 | https://github.com/docker/docker-py | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| dockerfile-parse | 2.0.1 | BSD-3-Clause | https://github.com/containerbuildsystem/dockerfile-parse | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| e2b | 2.52.0 | MIT | https://github.com/e2b-dev/e2b/tree/main/packages/python-sdk | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| easydict | 1.13 | LGPL-3.0 (only/or-later grant not explicit) | https://github.com/makinacorpus/easydict | Previous optional-serving/build review reference; not a fresh resolution | Not distributed — user-installed dependency |
| emerging-optimizers | v0.3.0 | Apache-2.0 | https://github.com/NVIDIA-NeMo/Emerging-Optimizers | Inherited recipe/source integration | Not distributed — user-installed dependency |
| emoji | 2.16.0 | BSD-3-Clause | https://github.com/carpedm20/emoji/ | External dependency: retained optional Tinker/IFBench integration | Not distributed — user-installed dependency |
| fast-hadamard-transform | e7706faf8d1c3b9f241e36860640ad1dac644ede | BSD-3-Clause | https://github.com/Dao-AILab/fast-hadamard-transform | Inherited recipe/source integration | Not distributed — user-installed dependency |
| fastapi | 0.142.2 | MIT | https://github.com/fastapi/fastapi | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| filelock | 4.0.10 | MIT | https://github.com/tox-dev/py-filelock | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| fla-core | 0.5.2 | MIT | https://github.com/fla-org/flash-linear-attention | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| flash-attn-4 | 4.0.0b19 | BSD-3-Clause | https://github.com/Dao-AILab/flash-attention | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| flash-linear-attention | v0.5.2 | MIT | https://github.com/fla-org/flash-linear-attention | Inherited recipe/source integration | Not distributed — user-installed dependency |
| flash-mla | main | MIT | https://github.com/deepseek-ai/FlashMLA | Inherited recipe/source integration | Not distributed — user-installed dependency |
| flashinfer-python | 0.6.17 | Apache-2.0 | https://github.com/flashinfer-ai/flashinfer | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| flashqla | 7c7dfe16416ad21b1d03258189fc8d3b8460ae06 | MIT | https://github.com/QwenLM/FlashQLA | Inherited recipe/source integration | Not distributed — user-installed dependency |
| Flask | 3.1.3 | BSD-3-Clause | https://github.com/pallets/flask/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| flask-cors | 6.0.5 | MIT | https://github.com/corydolphin/flask-cors | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| fonttools | 4.66.1 | MIT | http://github.com/fonttools/fonttools | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| frozenlist | 1.8.0 | Apache-2.0 | https://github.com/aio-libs/frozenlist | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| fsspec | 2026.6.0 | BSD-3-Clause | https://github.com/fsspec/filesystem_spec | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| gguf | 0.19.0 | MIT | https://github.com/ggml-org/llama.cpp | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| gitdb | 4.0.12 | BSD-3-Clause | https://github.com/gitpython-developers/gitdb | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| GitPython | 3.2.0 | BSD-3-Clause | https://pypi.org/project/GitPython/3.2.0/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| google-api-core | 2.40.0 | Apache-2.0 | https://github.com/googleapis/google-cloud-python | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| google-auth | 2.59.1 | Apache-2.0 | https://github.com/googleapis/google-cloud-python/tree/main/packages/google-auth | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| googleapis-common-protos | 1.75.5 | Apache-2.0 | https://github.com/googleapis/google-cloud-python/tree/main/packages/googleapis-common-protos | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| graphene | 3.4.3 | MIT | https://github.com/graphql-python/graphene | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| graphql-core | 3.2.13 | MIT | https://github.com/graphql-python/graphql-core | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| graphql-relay | 3.2.0 | MIT | https://github.com/graphql-python/graphql-relay-py | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| grpcio | 1.84.0 | Apache-2.0 | https://github.com/grpc/grpc | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| grpcio-health-checking | 1.81.1 | Apache-2.0 | https://grpc.io | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| grpcio-reflection | 1.81.1 | Apache-2.0 | https://grpc.io | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| grpcio-tools | 1.84.0 | Apache-2.0 | https://github.com/grpc/grpc/tree/master/tools/distrib/python/grpcio_tools | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| grpclib | 0.4.9 | BSD-3-Clause | https://github.com/vmagamedov/grpclib | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| gunicorn | 26.2.0 | MIT | https://gunicorn.org | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| h11 | 0.16.0 | MIT | https://github.com/python-hyper/h11 | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| h2 | 4.4.1 | MIT | https://github.com/python-hyper/h2/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| hf-xet | 1.6.0 | Apache-2.0 | https://github.com/huggingface/xet-core.git | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| hpack | 4.2.0 | MIT | https://github.com/python-hyper/hpack/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| httpcore | 1.0.9 | BSD-3-Clause | https://github.com/encode/httpcore | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| httpcore2 | 2.13.1 | BSD-3-Clause | https://github.com/pydantic/httpx2/blob/main/src/httpcore2 | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| httpx | 0.28.1 | BSD-3-Clause | https://github.com/encode/httpx | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| httpx2 | 2.13.1 | BSD-3-Clause | https://github.com/pydantic/httpx2 | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| huey | 3.4.0 | MIT | https://github.com/coleifer/huey | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| huggingface_hub | 1.33.0 | Apache-2.0 | https://github.com/huggingface/huggingface_hub | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| humming-kernels | 0.1.10 | Apache-2.0 | https://pypi.org/project/humming-kernels/0.1.10/ | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| hyperframe | 6.1.0 | MIT | https://github.com/python-hyper/hyperframe/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| hypothesis | 6.168.3 | MPL-2.0 | https://github.com/HypothesisWorks/hypothesis | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| idna | 3.20 | BSD-3-Clause | https://github.com/kjd/idna | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| ifbench | main | Apache-2.0 | https://github.com/allenai/IFBench | Inherited recipe/source integration | Not distributed — user-installed dependency |
| importlib_metadata | 9.0.1 | Apache-2.0 | https://github.com/python/importlib_metadata | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| iniconfig | 2.3.0 | MIT | https://github.com/pytest-dev/iniconfig | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| interegular | 0.3.3 | MIT | https://github.com/MegaIng/regex_intersections | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| ipython_pygments_lexers | 1.1.1 | BSD-3-Clause | https://github.com/ipython/ipython-pygments-lexers | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| itsdangerous | 2.2.0 | BSD-3-Clause | https://github.com/pallets/itsdangerous/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| Jinja2 | 3.1.6 | BSD-3-Clause | https://github.com/pallets/jinja/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| jiter | 0.17.0 | MIT | https://github.com/pydantic/jiter/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| joblib | 1.6.0 | BSD-3-Clause | https://github.com/joblib/joblib | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| jsonschema | 4.26.0 | MIT | https://github.com/python-jsonschema/jsonschema | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| jsonschema-specifications | 2025.9.1 | MIT | https://github.com/python-jsonschema/jsonschema-specifications | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| kernels | 0.14.1 | Apache-2.0 | https://pypi.org/project/kernels/0.14.1/ | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| kernels-data | 0.16.2 | Apache-2.0 | https://github.com/huggingface/kernels/tree/v0.16.2/kernels-data | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| kiwisolver | 1.5.1 | BSD-3-Clause | https://github.com/nucleic/kiwi | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| kubernetes_asyncio | 36.1.0 | Apache-2.0 | https://pypi.org/project/kubernetes_asyncio/36.1.0/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| lark | 1.3.1 | MIT | https://github.com/lark-parser/lark | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| linkify-it-py | 2.2.0 | MIT | https://github.com/tsutsu3/linkify-it-py | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| llguidance | 1.9.0 | MIT | https://github.com/microsoft/llguidance | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| loguru | 0.7.3 | MIT | https://github.com/Delgan/loguru | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| lxml | 6.1.3 | BSD-3-Clause | https://github.com/lxml/lxml | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| Mako | 1.4.3 | MIT | https://www.makotemplates.org/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| mamba-ssm | v2.3.1 | Apache-2.0 | https://github.com/state-spaces/mamba | Inherited recipe/source integration | Not distributed — user-installed dependency |
| Markdown | 3.11 | BSD-3-Clause | https://github.com/Python-Markdown/markdown | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| markdown-it-py | 4.2.0 | MIT | https://github.com/executablebooks/markdown-it-py | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| MarkupSafe | 3.0.4 | BSD-3-Clause | https://github.com/pallets/markupsafe/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| matplotlib | 3.11.2 | Matplotlib license with third-party notices (including MIT and Apache-2.0) | https://github.com/matplotlib/matplotlib | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| mbridge | 89eb10887887bc74853f89a4de258c0702932a1c | BSD-3-Clause AND Apache-2.0 AND MIT | https://github.com/ISEEKYAN/mbridge | Inherited recipe/source integration | Not distributed — user-installed dependency |
| mcp | 2.3.0 | MIT | https://github.com/modelcontextprotocol/python-sdk | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| mcp-types | 2.3.0 | MIT | https://github.com/modelcontextprotocol/python-sdk | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| mdit-py-plugins | 0.6.1 | MIT | https://github.com/executablebooks/mdit-py-plugins | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| mdurl | 0.1.2 | MIT | https://github.com/executablebooks/mdurl | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| megatron-energon | main | BSD-3-Clause | https://github.com/NVIDIA/Megatron-Energon | Inherited recipe/source integration | Not distributed — user-installed dependency |
| memray | 1.20.0 | Apache-2.0 | https://github.com/bloomberg/memray | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| MindSpeed | fc63de5c48426dd019c3b3f39e65f5bdf56e4086 | Conflicting terms: BSD-3-Clause / Apache-2.0 / MIT; noncommercial README | https://gitcode.com/Ascend/MindSpeed | Inherited optional NPU guide; not the OLMo-core training path | Not distributed — user-installed dependency |
| mistral_common | 1.12.0 | Apache-2.0 | https://pypi.org/project/mistral_common/1.12.0/ | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| ml_dtypes | 0.6.0 | Apache-2.0 | https://pypi.org/project/ml_dtypes/0.6.0/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| mlflow | 3.16.1 | Apache-2.0 | https://pypi.org/project/mlflow/3.16.1/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| mlflow-skinny | 3.16.1 | Apache-2.0 | https://pypi.org/project/mlflow-skinny/3.16.1/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| mlflow-tracing | 3.16.1 | Apache-2.0 | https://pypi.org/project/mlflow-tracing/3.16.1/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| modal | 1.6.1 | Apache-2.0 | https://github.com/modal-labs/modal-client | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| modelscope | 1.40.1 | Apache-2.0 | https://github.com/modelscope/modelscope | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| modelscope-hub | 0.4.5 | Apache-2.0 | https://pypi.org/project/modelscope-hub/0.4.5/ | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| mooncake | 0.3.12.post1 (ROCm recipe); wheel-selected CUDA variant | Apache-2.0 | https://github.com/kvcache-ai/Mooncake | Inherited recipe/source integration | Not distributed — user-installed dependency |
| mpmath | 1.3.0 | BSD-3-Clause | https://github.com/fredrik-johansson/mpmath | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| msgpack | 1.2.3 | Apache-2.0 | https://github.com/msgpack/msgpack-python/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| multi-storage-client | main | Apache-2.0 | https://github.com/NVIDIA/multi-storage-client | Inherited recipe/source integration | Not distributed — user-installed dependency |
| multidict | 6.9.1 | Apache-2.0 | https://github.com/aio-libs/multidict | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| multiprocess | 0.70.19 | BSD-3-Clause | https://github.com/uqfoundation/multiprocess | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| narwhals | 2.26.0 | MIT | https://github.com/narwhals-dev/narwhals | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| nccl-tests | ae98985f5599617be94042f4aa3637d10014ce89 | BSD-3-Clause | https://github.com/NVIDIA/nccl-tests | Inherited recipe/source integration | Not distributed — user-installed dependency |
| nccl4py | 0.6.0 | Apache-2.0 | https://github.com/NVIDIA/nccl | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| networkx | 3.7 | BSD-3-Clause | https://github.com/networkx/networkx | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://pypi.org/project/numpy/2.5.3/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| nvdlfw-inspect | main | Apache-2.0 | https://github.com/NVIDIA/nvidia-dlfw-inspect | Inherited recipe/source integration | Not distributed — user-installed dependency |
| nvidia-cublas | 13.1.1.3 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| nvidia-cuda-cccl | 13.3.4.3.1 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone | Previous optional-serving/build review reference; not a fresh resolution | Not distributed — user-installed dependency |
| nvidia-cuda-crt | 13.4.92 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone | Previous optional-serving/build review reference; not a fresh resolution | Not distributed — user-installed dependency |
| nvidia-cuda-cupti | 13.0.85 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| nvidia-cuda-nvcc | 13.4.92 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone | Previous optional-serving/build review reference; not a fresh resolution | Not distributed — user-installed dependency |
| nvidia-cuda-nvdisasm | 13.4.92 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone | Previous optional-serving/build review reference; not a fresh resolution | Not distributed — user-installed dependency |
| nvidia-cuda-nvrtc | 13.0.88 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| nvidia-cuda-runtime | 13.0.96 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| nvidia-cudnn-cu13 | 9.24.0.43 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| nvidia-cudnn-frontend | 1.30.0 | Apache-2.0 AND MIT | https://github.com/NVIDIA/cudnn-frontend | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| nvidia-cufft | 12.0.0.61 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| nvidia-cufile | 1.15.1.6 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| nvidia-curand | 10.4.0.35 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| nvidia-cusolver | 12.0.4.66 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| nvidia-cusparse | 12.6.3.3 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| nvidia-cusparselt-cu13 | 0.8.1 | NVIDIA Proprietary Software | https://developer.nvidia.com/cusparselt | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| nvidia-cutlass-dsl | 4.6.2 | Other/Proprietary License | https://github.com/NVIDIA/cutlass | Previous optional-serving/build review reference; not a fresh resolution | Not distributed — user-installed dependency |
| nvidia-cutlass-dsl-libs-base | 4.6.2 | Other/Proprietary License | https://github.com/NVIDIA/cutlass | Previous optional-serving/build review reference; not a fresh resolution | Not distributed — user-installed dependency |
| nvidia-cutlass-dsl-libs-core | 4.6.2 | Other/Proprietary License | https://github.com/NVIDIA/cutlass | Previous optional-serving/build review reference; not a fresh resolution | Not distributed — user-installed dependency |
| nvidia-cutlass-dsl-libs-cu12 | 4.6.2 | Other/Proprietary License | https://github.com/NVIDIA/cutlass | Previous optional-serving/build review reference; not a fresh resolution | Not distributed — user-installed dependency |
| nvidia-cutlass-dsl-libs-cu13 | 4.6.2 | Other/Proprietary License | https://github.com/NVIDIA/cutlass | Previous optional-serving/build review reference; not a fresh resolution | Not distributed — user-installed dependency |
| nvidia-mathdx | 25.6.0 | Other/Proprietary License | https://developer.nvidia.com/cufftdx-downloads | Previous optional-serving/build review reference; not a fresh resolution | Not distributed — user-installed dependency |
| nvidia-ml-py | 13.615.71 | BSD-3-Clause | https://forums.developer.nvidia.com | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| nvidia-modelopt | main | Apache-2.0 | https://github.com/NVIDIA/Model-Optimizer | Inherited recipe/source integration | Not distributed — user-installed dependency |
| nvidia-nccl-cu13 | 2.30.7 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| nvidia-nvjitlink | 13.4.92 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| nvidia-nvshmem-cu13 | 3.4.5 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| nvidia-nvtx | 13.0.85 | Apache-2.0 | https://developer.nvidia.com/cuda-zone | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| nvidia-nvvm | 13.4.92 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone | Previous optional-serving/build review reference; not a fresh resolution | Not distributed — user-installed dependency |
| nvidia-resiliency-ext | 0.6.0 | Apache-2.0 | https://github.com/NVIDIA/nvidia-resiliency-ext | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| olmo-core-runtime | 4f00ddcb475c3c9dc6c7005217b7572b5bf7582f | Apache-2.0 with third-party notices | https://github.com/allenai/OLMo-core | Inherited recipe/source integration | Not distributed — user-installed dependency |
| olmo-sglang-runtime | b3a795f33e048dac401c89bb5823381146181ac6 | Apache-2.0 with third-party notices | https://github.com/allenai/olmo-sglang | Inherited recipe/source integration | Not distributed — user-installed dependency |
| omegaconf | 2.3.1 | BSD-3-Clause | https://github.com/omry/omegaconf | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| onnx | 1.23.1 | Apache-2.0 | https://github.com/onnx/onnx | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| onnx-ir | 1.0.0 | Apache-2.0 | https://github.com/onnx/ir-py | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| onnxscript | 0.7.2 | MIT | https://github.com/microsoft/onnxscript | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| open-instruct | c20ab00a8175b35517629b14ea2404b0be924ac6 | Apache-2.0 | https://github.com/allenai/open-instruct | Inherited recipe/source integration | Not distributed — user-installed dependency |
| openai | 3.24.0 | Apache-2.0 | https://github.com/openai/openai-python | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| openai-harmony | 0.0.4 | Apache-2.0 | https://pypi.org/project/openai-harmony/0.0.4/ | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| opencensus | 0.11.4 | Apache-2.0 | https://github.com/census-instrumentation/opencensus-python | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| opencensus-context | 0.1.3 | Apache-2.0 | https://github.com/census-instrumentation/opencensus-python/tree/master/context/opencensus-context | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| opentelemetry-api | 1.45.0 | Apache-2.0 | https://github.com/open-telemetry/opentelemetry-python | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| opentelemetry-exporter-http-transport | 0.66b0 | Apache-2.0 | https://github.com/open-telemetry/opentelemetry-python | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| opentelemetry-exporter-otlp-common | 0.66b0 | Apache-2.0 | https://github.com/open-telemetry/opentelemetry-python | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| opentelemetry-exporter-otlp-proto-common | 1.45.0 | Apache-2.0 | https://github.com/open-telemetry/opentelemetry-python | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| opentelemetry-exporter-otlp-proto-http | 1.45.0 | Apache-2.0 | https://github.com/open-telemetry/opentelemetry-python | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| opentelemetry-exporter-prometheus | 0.66b0 | Apache-2.0 | https://github.com/open-telemetry/opentelemetry-python | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| opentelemetry-proto | 1.45.0 | Apache-2.0 | https://github.com/open-telemetry/opentelemetry-python | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| opentelemetry-sdk | 1.45.0 | Apache-2.0 | https://github.com/open-telemetry/opentelemetry-python | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| opentelemetry-semantic-conventions | 0.66b0 | Apache-2.0 | https://github.com/open-telemetry/opentelemetry-python | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| orjson | 3.12.0 | MPL-2.0 AND (Apache-2.0 OR MIT) | https://pypi.org/project/orjson/3.12.0/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| outlines | 0.1.11 | Apache-2.0 | https://pypi.org/project/outlines/0.1.11/ | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| outlines_core | 0.1.26 | Apache-2.0 | https://pypi.org/project/outlines_core/0.1.26/ | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| pandas | 3.0.6 | BSD-3-Clause | https://pypi.org/project/pandas/3.0.6/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| partial-json-parser | 0.2.1.1.post7 | MIT | https://pypi.org/project/partial-json-parser/0.2.1.1.post7/ | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| pillow | 12.3.0 | MIT-CMU | https://github.com/python-pillow/Pillow | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| platformdirs | 4.12.3 | MIT | https://github.com/tox-dev/platformdirs | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| pluggy | 1.6.0 | MIT | https://pypi.org/project/pluggy/1.6.0/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| polars | 1.42.1 | MIT | https://github.com/pola-rs/polars | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| polars-runtime-32 | 1.42.1 | MIT | https://github.com/pola-rs/polars | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| prettytable | 3.18.0 | BSD-3-Clause | https://github.com/prettytable/prettytable | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| prometheus_client | 0.26.0 | Apache-2.0 AND BSD-2-Clause | https://github.com/prometheus/client_python | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| propcache | 0.5.4 | Apache-2.0 | https://github.com/aio-libs/propcache | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| proto-plus | 1.29.0 | Apache-2.0 | https://github.com/googleapis/google-cloud-python | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| protobuf | 7.36.2 | BSD-3-Clause | https://developers.google.com/protocol-buffers/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| protobuf-py | 0.1.1 | Apache-2.0 | https://github.com/bufbuild/protobuf-py | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| protobuf-py-ext | 0.1.1 | Apache-2.0 | https://github.com/bufbuild/protobuf-py | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| psutil | 7.2.2 | BSD-3-Clause | https://github.com/giampaolo/psutil | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| psycopg | 3.3.6 | LGPL-3.0-only | https://github.com/psycopg/psycopg | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| psycopg-binary | 3.3.6 | LGPL-3.0-only | https://github.com/psycopg/psycopg | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| py-spy | 0.4.2 | MIT | https://github.com/benfred/py-spy | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| pyarrow | 25.0.1 | Apache-2.0 | https://github.com/apache/arrow | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| pyasn1 | 0.6.4 | BSD-2-Clause | https://github.com/pyasn1/pyasn1 | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| pyasn1_modules | 0.4.2 | BSD-2-Clause | https://github.com/pyasn1/pyasn1-modules | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| pybase64 | 1.5.0 | BSD-2-Clause | https://github.com/mayeut/pybase64 | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| pycountry | 26.2.16 | LGPL-2.1-only | https://github.com/pycountry/pycountry | Previous optional-serving/build review reference; not a fresh resolution | Not distributed — user-installed dependency |
| pycparser | 3.0 | BSD-3-Clause | https://github.com/eliben/pycparser | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| pycryptodomex | 3.23.0 | BSD-2-Clause and public-domain portions | https://github.com/Legrandin/pycryptodome/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| pydantic | 2.13.5 | MIT | https://github.com/pydantic/pydantic | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| pydantic-extra-types | 2.11.1 | MIT | https://github.com/pydantic/pydantic-extra-types | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| pydantic_core | 2.46.5 | MIT | https://github.com/pydantic/pydantic/tree/main/pydantic-core | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| pyelftools | 0.33 | Public domain | https://github.com/eliben/pyelftools.git | Previous optional-serving/build review reference; not a fresh resolution | Not distributed — user-installed dependency |
| Pygments | 2.21.0 | BSD-2-Clause | https://github.com/pygments/pygments | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| PyJWT | 2.15.1 | MIT | https://github.com/jpadilla/pyjwt | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| pylatexenc | 2.11 | MIT | https://github.com/phfaist/pylatexenc | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| pyparsing | 3.3.3 | MIT | https://github.com/pyparsing/pyparsing.git | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| pyqwest | 0.10.0 | MIT | https://github.com/curioswitch/pyqwest.git | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| pytest | 9.1.1 | MIT | https://github.com/pytest-dev/pytest | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| pytest-asyncio | 1.4.0 | Apache-2.0 | https://github.com/pytest-dev/pytest-asyncio | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| python-dateutil | 2.9.0.post0 | Apache-2.0 OR BSD-3-Clause | https://github.com/dateutil/dateutil | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| python-discovery | 1.6.1 | MIT | https://github.com/tox-dev/python-discovery | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| python-dotenv | 1.2.4 | BSD-3-Clause | https://github.com/theskumar/python-dotenv | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| python-multipart | 0.0.32 | Apache-2.0 | https://github.com/Kludex/python-multipart | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| PyYAML | 6.0.3 | MIT | https://github.com/yaml/pyyaml | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| quack-kernels | 0.6.4 | Apache-2.0 | https://pypi.org/project/quack-kernels/0.6.4/ | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| qwen-vl-utils | 0.0.14 | Apache-2.0 | https://github.com/QwenLM/Qwen2-VL.git | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| ray | 2.59.0 | Apache-2.0 | https://github.com/ray-project/ray | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| referencing | 0.37.0 | MIT | https://github.com/python-jsonschema/referencing | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python | https://github.com/mrabarnett/mrab-regex | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| requests | 2.34.2 | Apache-2.0 | https://github.com/psf/requests | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| rich | 15.0.0 | MIT | https://github.com/Textualize/rich | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| ring-flash-attn | 0.1.8 | MIT | https://github.com/zhuzilin/ring-flash-attention | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| rocm-systems | 746c7b3c9e78b094389e19499919fdd43a0b6a90 | BSD-3-Clause | https://github.com/ROCm/rocm-systems | Inherited recipe/source integration | Not distributed — user-installed dependency |
| rpds-py | 2026.6.3 | MIT | https://github.com/crate-py/rpds | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| safetensors | 0.8.0 | Apache-2.0 | https://github.com/huggingface/safetensors | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| scikit-learn | 1.9.1 | BSD-3-Clause | https://pypi.org/project/scikit-learn/1.9.1/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| scipy | 1.18.1 | BSD-3-Clause; external wheel includes additional third-party license terms | https://github.com/scipy/scipy | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| setproctitle | 1.3.8 | BSD-3-Clause | https://github.com/dvarrazzo/py-setproctitle | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| setuptools | 84.0.0 | MIT | https://github.com/pypa/setuptools | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| sgl-deep-ep | 0.1.0 | Apache-2.0 | https://pypi.org/project/sgl-deep-ep/0.1.0/ | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| sgl-deep-gemm | 0.1.5.post3 | Apache-2.0 | https://github.com/sgl-project/DeepGEMM/tree/release | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| sgl-kernel-npu | 2026.05.01 | MIT | https://github.com/sgl-project/sgl-kernel-npu | Inherited recipe/source integration | Not distributed — user-installed dependency |
| sgl-router-for-miles | main | Apache-2.0 | https://github.com/radixark/sgl-router-for-miles | Inherited recipe/source integration | Not distributed — user-installed dependency |
| sglang-kernel | 0.4.6.post1 | Apache-2.0 | https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/aot | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| sglang-router | 0.3.2 | Apache-2.0 | https://pypi.org/project/sglang-router/0.3.2/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| shellingham | 1.5.4 | ISC | https://github.com/sarugaku/shellingham | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| six | 1.17.0 | MIT | https://github.com/benjaminp/six | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| skops | 0.16.0 | MIT | http://github.com/skops-dev/skops | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| smart_open | 8.0.2 | MIT | https://github.com/piskvorky/smart_open | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| smg-grpc-proto | 0.4.22 | Apache-2.0 | https://github.com/smg-project/smg | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| smg-grpc-servicer | 0.13.0 | Apache-2.0 | https://github.com/smg-project/smg | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| smmap | 5.0.3 | BSD-3-Clause | https://github.com/gitpython-developers/smmap | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| sniffio | 1.3.1 | MIT OR Apache-2.0 | https://github.com/python-trio/sniffio | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| sortedcontainers | 2.4.0 | Apache-2.0 | http://www.grantjenks.com/docs/sortedcontainers/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| SQLAlchemy | 2.1.3 | MIT | https://github.com/sqlalchemy/sqlalchemy | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| sqlparse | 0.6.0 | BSD-3-Clause | https://github.com/andialbrecht/sqlparse | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| sse-starlette | 3.5.0 | BSD-3-Clause | https://github.com/sysid/sse-starlette | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| starlette | 1.7.0 | BSD-3-Clause | https://github.com/Kludex/starlette | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| syllapy | 0.8.0 | MIT | https://github.com/mholtzscher/syllapy | External dependency: retained optional Tinker/IFBench integration | Not distributed — user-installed dependency |
| sympy | 1.14.0 | BSD-3-Clause | https://github.com/sympy/sympy | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| synchronicity | 0.12.6 | Apache-2.0 | https://pypi.org/project/synchronicity/0.12.6/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| tensorboard | 2.21.0 | Apache-2.0 | https://github.com/tensorflow/tensorboard | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| tensorboard-data-server | 0.7.2 | Apache-2.0 | https://github.com/tensorflow/tensorboard/tree/master/tensorboard/data/server | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| textual | 8.2.8 | MIT | https://github.com/Textualize/textual | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| threadpoolctl | 3.7.0 | BSD-3-Clause | https://pypi.org/project/threadpoolctl/3.7.0/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| tilelang | 0.1.11 | MIT | https://pypi.org/project/tilelang/0.1.11/ | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| tilelang-cuda | 7e3bbf3703e769479757eb7ecec2f2626f3cd0a8 | MIT with additional historical collaboration note | https://github.com/tile-ai/tilelang | Inherited recipe/source integration | Not distributed — user-installed dependency |
| tinker | 0.26.2 | Apache-2.0 | https://github.com/thinking-machines-lab/tinker | External dependency: retained optional Tinker/IFBench integration | Not distributed — user-installed dependency |
| tokenizers | 0.22.2 | Apache-2.0 | https://github.com/huggingface/tokenizers | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| tokenspeed-mla | 0.1.8 | MIT | https://github.com/lightseekorg/tokenspeed | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| tokenspeed-triton | 3.8.10.post20260920 | MIT | https://github.com/triton-lang/triton/ | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| toml | 0.10.2 | MIT | https://github.com/uiri/toml | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| torch | 2.14.1 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | https://github.com/pytorch/pytorch | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| torch-memory-saver-cuda | b5588e83de86412a48689a6583a4b567e75f7acc | MIT | https://github.com/fzyzcjy/torch_memory_saver | Inherited recipe/source integration | Not distributed — user-installed dependency |
| torch-memory-saver-rocm | 06caa534822c5b980b61733a5d79ac731f9a5f9c | MIT | https://github.com/fzyzcjy/torch_memory_saver | Inherited recipe/source integration | Not distributed — user-installed dependency |
| torch-npu | c4194988d3c830375abf08e01d5bec3fe6a86406 | BSD-3-Clause | https://github.com/Ascend/pytorch | Inherited recipe/source integration | Not distributed — user-installed dependency |
| torch_c_dlpack_ext | 0.1.5 | Apache-2.0 | https://pypi.org/project/torch_c_dlpack_ext/0.1.5/ | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| torch_memory_saver | 0.0.10 | MIT | https://pypi.org/project/torch_memory_saver/0.0.10/ | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| torchcodec | 0.15.0+cu130 | BSD-3-Clause | https://pypi.org/project/torchcodec/0.15.0+cu130/ | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| torchft-nightly | 2026.4.3 | BSD-3-Clause | https://github.com/pytorch/torchft | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | https://pypi.org/project/tqdm/4.70.1/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| transformer-engine-rocm | a365f2deb555e681bf5b259e344b2bd1fbb4f87c | Apache-2.0 AND MIT | https://github.com/ROCm/TransformerEngine | Inherited recipe/source integration | Not distributed — user-installed dependency |
| transformers | 5.12.1 | Apache-2.0 | https://github.com/huggingface/transformers | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| triton | 3.8.0 | MIT | https://github.com/triton-lang/triton/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| truststore | 0.10.4 | MIT | https://github.com/sethmlarson/truststore | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| typer | 0.27.2 | MIT | https://github.com/fastapi/typer | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| types-certifi | 2021.10.8.3 | Apache-2.0 | https://github.com/python/typeshed | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| types-toml | 0.10.8.20260518 | Apache-2.0 | https://github.com/python/typeshed | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| typing-inspection | 0.4.4 | MIT | https://github.com/pydantic/typing-inspection | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| typing_extensions | 4.16.0 | PSF-2.0 | https://github.com/python/typing_extensions | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| urllib3 | 2.8.0 | MIT | https://pypi.org/project/urllib3/2.8.0/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| uvicorn | 0.54.0 | BSD-3-Clause | https://github.com/Kludex/uvicorn | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| uvloop | 0.22.1 | MIT | https://pypi.org/project/uvloop/0.22.1/ | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| virtualenv | 21.14.5 | MIT | https://github.com/pypa/virtualenv | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| wandb | 0.30.0 | MIT | https://github.com/wandb/wandb | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| watchfiles | 1.3.0 | MIT | https://github.com/samuelcolvin/watchfiles | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| wcmatch | 11.0.1 | MIT | https://github.com/facelessuser/wcmatch | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| wcwidth | 0.9.1 | MIT | https://github.com/jquast/wcwidth | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| Werkzeug | 3.1.9 | BSD-3-Clause | https://github.com/pallets/werkzeug/ | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| wrapt | 2.5.0 | BSD-2-Clause | https://github.com/GrahamDumpleton/wrapt | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| xgrammar | 0.2.1 | Apache-2.0 | https://xgrammar.mlc.ai/ | External dependency: optional serving version recorded in dependency review | Not distributed — user-installed dependency |
| xxhash | 3.7.1 | BSD-2-Clause | https://github.com/ifduyue/python-xxhash | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| yarl | 1.25.1 | Apache-2.0 | https://github.com/aio-libs/yarl | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| zipp | 4.1.1 | MIT | https://github.com/jaraco/zipp | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |
| zstandard | 0.25.0 | BSD-3-Clause | https://github.com/indygreg/python-zstandard | External dependency: current base/declared-extras resolution | Not distributed — user-installed dependency |

## Previous submission reconciliation

The [machine-readable review](third_party/dependency-review.json) accounts for all 224 original submission rows, retaining their submitted versions and licenses alongside the current disposition. Historical rows do not assert that an old version is still installed or supported. The removed entries are summarized below.

| Previously submitted component | Submitted version | Current disposition |
| --- | --- | --- |
| aws-bedrock-token-generator | 1.1.0 | Former example-only inventory entry; example installation removed |
| camel-ai | 0.2.90 | Former example-only inventory entry; example installation removed |
| DeepSeek-V3.2 encoding (adapted source) | Via SGLang 0.5.14.dev37+gf8cfad3; original revision not recorded | Removed source/install surface |
| flashinfer-python | 0.6.17 | Adapted fake-QAT code removed; external FlashInfer remains user-installed (not distributed). |
| gepa | 0.1.4 | Former example-only inventory entry; example installation removed |
| google_search_results | 2.4.2 | Former example-only inventory entry; example installation removed |
| griffe | 1.15.0 | Former example-only inventory entry; example installation removed |
| httpx-sse | 0.4.3 | Former example-only inventory entry; example installation removed |
| latex2sympy2_extended | 1.11.0 | Former example-only inventory entry; example installation removed |
| markdownify | 1.2.3 | Former example-only inventory entry; example installation removed |
| math-verify | 0.9.0 | Former example-only inventory entry; example installation removed |
| MindSpeed | fc63de5c48426dd019c3b3f39e65f5bdf56e4086 (patched) | MindSpeed patch removed; external installation remains documented in the inherited NPU guide (not distributed). |
| openai-agents | 0.4.2 | Former example-only inventory entry; example installation removed |
| opentelemetry-instrumentation-threading | 0.66b0 | Former example-only inventory entry; example installation removed |
| prime-pydantic-config | 0.4.3 | Former example-only inventory entry; example installation removed |
| prime-sandboxes | 0.4.1 | Former example-only inventory entry; example installation removed |
| prime-tunnel | 0.1.11 | Former example-only inventory entry; example installation removed |
| ProRL-Agent-Server (adapted source) | Version not recorded in source attribution | Removed source/install surface |
| pydantic-settings | 2.15.0 | Former example-only inventory entry; example installation removed |
| renderers | 0.1.10 | Former example-only inventory entry; example installation removed |
| slack_bolt | 1.30.0 | Former example-only inventory entry; example installation removed |
| slack_sdk | 3.44.1 | Former example-only inventory entry; example installation removed |
| strands-agents | 1.57.2 | Former example-only inventory entry; example installation removed |
| strands-agents-tools | 0.8.9 | Former example-only inventory entry; example installation removed |
| strands-sglang | 0.6.1 | Former example-only inventory entry; example installation removed |
| TileKernels | 1.0.0 (patched) | Removed source/install surface |
| tilelang | 0.1.11 | TileLang source patches removed; external CUDA recipe still selects 0.1.14 (not distributed). Historical runtime versions are reference evidence. |
| tilelang | 0.1.14 (CUDA); ROCm patch variant | TileLang source patches removed; external CUDA recipe still selects 0.1.14 (not distributed). Historical runtime versions are reference evidence. |
| tinker_cookbook | 0.5.8.dev1+g1f962eda3 | Former example-only inventory entry; example installation removed |
| tml-renderers | 0.1.0 | Former example-only inventory entry; example installation removed |
| tomli_w | 1.2.0 | Former example-only inventory entry; example installation removed |
| types-requests | 2.33.0.20260906 | Former example-only inventory entry; example installation removed |
| verifiers | 0.2.0 | Former example-only inventory entry; example installation removed |
