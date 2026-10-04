---
title: DeepSeek-V4 Pro
description: Launch recipe for DeepSeek-V4-Pro (1.6 T) — V4-family architecture at Pro scale.
---

This fork omits the GLM-5 and DeepSeek-V3.2/V4 training plugins and their adapted
TileLang kernels. The training launchers described upstream are not included.

See the [upstream model guide](https://miles.radixark.com/docs/models/deepseek/deepseek-v4-pro)
for the original recipes. External teacher serving is independent of these training plugins.
