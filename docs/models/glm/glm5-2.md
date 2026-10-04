---
title: GLM-5.2
description: Launch recipe for GLM-5.2 (744 B / 40 B active) — FP8 KV cache, TIS, 16+ node config.
---

This fork omits the GLM-5 and DeepSeek-V3.2/V4 training plugins and their adapted
TileLang kernels. The training launchers described upstream are not included.

See the [upstream model guide](https://miles.radixark.com/docs/models/glm/glm5-2)
for the original recipes. External teacher serving is independent of these training plugins.
