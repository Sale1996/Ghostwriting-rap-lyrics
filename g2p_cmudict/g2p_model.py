import torch
import torch.nn as nn
from torch.autograd import Variable

from g2p_cmudict.beam_search import Beam
from g2p_cmudict.decoder import Decoder
from g2p_cmudict.encoder import Encoder


class G2P(nn.Module):

    def __init__(self, config):
        super(G2P, self).__init__()
        self.encoder = Encoder(config.g_size, config.d_embed,
                               config.d_hidden)
        self.decoder = Decoder(config.p_size, config.d_embed,
                               config.d_hidden)
        self.config = config

    def forward(self, g_seq, p_seq=None):
        o, h, c = self.encoder(g_seq, self.config.cuda)
        context = o.transpose(0, 1) if self.config.attention else None
        if p_seq is not None:  # not generate
            return self.decoder(p_seq, h, c, context)
        else:
            assert g_seq.size(1) == 1  # make sure batch_size = 1
            return self._generate(h, c, context)

    def _generate(self, h, c, context):
        beam = Beam(self.config.beam_size, cuda=self.config.cuda)
        # Make a beam_size batch.
        h = h.expand(beam.size, h.size(1))
        c = c.expand(beam.size, c.size(1))
        context = context.expand(beam.size, context.size(1), context.size(2))

        for i in range(self.config.max_len):  # max_len = 20
            x = beam.get_current_state()
            o, h, c = self.decoder(Variable(x.unsqueeze(0)), h, c, context)
            if beam.advance(o.data.squeeze(0)):
                break
            h.data.copy_(h.data.index_select(0, beam.get_current_origin()))
            c.data.copy_(c.data.index_select(0, beam.get_current_origin()))
        tt = torch.cuda if self.config.cuda else torch
        return Variable(tt.LongTensor(beam.get_hyp(0)))


