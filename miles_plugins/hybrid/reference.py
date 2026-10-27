"""Reference head with FP32 logits from the same stored model weights."""

import torch
from torch import nn
from torch.nn import functional as F


class FP32Head(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.weight = head.weight

    def forward(self, hidden):
        return F.linear(hidden.float(), self.weight.float())


@torch.inference_mode()
def cached_logprobs(model, prompt, generated, first_position=None):
    """Teacher-force the served continuation through the HF recurrent decode path.

    Return predictions starting at first_position (a target-token position).
    Default to the first generated token; retain prompt suffix scores when asked.
    """
    first_position = len(prompt) if first_position is None else first_position
    if not 1 <= first_position <= len(prompt) or not generated:
        raise ValueError("Expected a nonempty continuation and a valid target position")
    inputs = torch.tensor([prompt], device=model.device)
    output = model(inputs, use_cache=True, logits_to_keep=len(prompt) - first_position + 1)
    scores = [output.logits[0].float().log_softmax(-1)]
    cache = output.past_key_values
    for token in generated[:-1]:
        inputs = torch.tensor([[token]], device=model.device)
        output = model(inputs, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        scores.append(output.logits[0].float().log_softmax(-1))
    return torch.cat(scores)
