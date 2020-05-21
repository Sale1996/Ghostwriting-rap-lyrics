import torch
import torch.nn as nn
from torch.autograd import Variable


class Encoder(nn.Module):

    def __init__(self, vocab_size, d_embed, d_hidden):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.lstm = nn.LSTMCell(d_embed, d_hidden)
        self.d_hidden = d_hidden

    def forward(self, x_seq, cuda=False):
        o = []
        e_seq = self.embedding(x_seq)  # seq x batch x dim
        tt = torch.cuda if cuda else torch  # use cuda tensor or not
        # create initial hidden state and initial cell state
        h = Variable(tt.FloatTensor(e_seq.size(1), self.d_hidden).zero_())
        c = Variable(tt.FloatTensor(e_seq.size(1), self.d_hidden).zero_())

        for e in e_seq.chunk(e_seq.size(0), 0):
            e = e.squeeze(0)
            h, c = self.lstm(e, (h, c))
            o.append(h)
        return torch.stack(o, 0), h, c
