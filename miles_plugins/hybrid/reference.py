"""Reference head with FP32 logits from the same stored model weights."""

from torch import nn
from torch.nn import functional as F


class FP32Head(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.weight = head.weight

    def forward(self, hidden):
        return F.linear(hidden.float(), self.weight.float())
