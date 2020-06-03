import torch
import torch.nn as nn
import torch.nn.functional as F
from g2p_cmudict.attention import Attention


class Decoder(nn.Module):

    def __init__(self, vocab_size, d_embed, d_hidden):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.lstm = nn.LSTMCell(d_embed, d_hidden)
        self.attn = Attention(d_hidden)
        self.linear = nn.Linear(d_hidden, vocab_size)

    def forward(self, x_seq, h, c, context=None):
        o = []
        e_seq = self.embedding(x_seq)
        for e in e_seq.chunk(e_seq.size(0), 0):
            e = e.squeeze(0)
            h, c = self.lstm(e, (h, c))
            o.append(self.attn(h, context))
        o = torch.stack(o, 0)
        o = self.linear(o.view(-1, h.size(1)))
        return F.log_softmax(o, dim=1).view(x_seq.size(0), -1, o.size(1)), h, c