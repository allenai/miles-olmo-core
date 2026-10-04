---
title: DeepSeek-V4.1 Flash
description: Launch recipe for DeepSeek-V4.1 with radixark/miles:deepseek-v41 — BF16 train / BF16 rollout, colocated, 4-node GB300 (16 GPUs) with optimizer state streamed to NVMe.
---

This fork omits the GLM-5 and DeepSeek-V3.2/V4 training plugins and their adapted
TileLang kernels. The training launchers described upstream are not included.

See the [upstream model guide](https://miles.radixark.com/docs/models/deepseek/deepseek-v4-1-flash)
for the original recipes. External teacher serving is independent of these training plugins.
