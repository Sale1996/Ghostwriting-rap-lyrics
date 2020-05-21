import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Dot global attention from https://arxiv.org/abs/1508.04025"""

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, x, context=None):
        if context is None:
            return x
        assert x.size(0) == context.size(0)  # x: batch x dim
        assert x.size(1) == context.size(2)  # context: batch x seq x dim
        attn = F.softmax(context.bmm(x.unsqueeze(2)).squeeze(2), dim=1)
        weighted_context = attn.unsqueeze(1).bmm(context).squeeze(1)
        o = self.linear(torch.cat((x, weighted_context), 1))
        return F.tanh(o)