---
title: DeepSeek-V3.2
description: Launch recipe for DeepSeek-V3.2 (671 B total / 37 B active) — BF16 training, NSA rollout, 8 training nodes and up.
---

This fork omits the GLM-5 and DeepSeek-V3.2/V4 training plugins and their adapted
TileLang kernels. The training launchers described upstream are not included.

See the [upstream model guide](https://miles.radixark.com/docs/models/deepseek/deepseek-v3-2)
for the original recipes. External teacher serving is independent of these training plugins.
