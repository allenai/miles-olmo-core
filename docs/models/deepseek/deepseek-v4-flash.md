---
title: DeepSeek-V4 Flash
description: Launch recipe for DeepSeek-V4-Flash (284 B) — FP8 rollout / BF16 train, 8-node H200 (64 GPUs).
---

This fork omits the GLM-5 and DeepSeek-V3.2/V4 training plugins and their adapted
TileLang kernels. The training launchers described upstream are not included.

See the [upstream model guide](https://miles.radixark.com/docs/models/deepseek/deepseek-v4-flash)
for the original recipes. External teacher serving is independent of these training plugins.
