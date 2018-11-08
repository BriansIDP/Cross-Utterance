import torch.nn as nn
from torch import cat
from torch.autograd import Variable
from linear_nce import linear_nce

class L2RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nutt, naux, nhid, nlayers, dropout=0.5, tie_weights=False, loss_type='ce', unigram_prob = None, num_noise=25, reset=0):
        super(L2RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.compressor = nn.Linear(nutt, naux)
        self.compressReLU = nn.ReLU()
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp+naux, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.loss_type = loss_type
        self.num_noise = num_noise # for nce
        if loss_type == 'nce':
            self.decoder = linear_nce(nhid, ntoken, unigram_prob)
        else:
            self.decoder = nn.Linear(nhid, ntoken)
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.reset = reset
        self.mode = 'train'

    def set_mode(self, m):
        self.mode = m

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.compressor.bias.data.zero_()
        self.compressor.weight.data.uniform_(-initrange, initrange)
        if self.loss_type != 'nce':
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, auxiliary, hidden, separate=0, eosidx = 0, target=None):
        emb = self.drop(self.encoder(input))
        auxiliary_in = self.compressor(auxiliary.view(auxiliary.size(0)*auxiliary.size(1), auxiliary.size(2)))
        auxiliary_in = self.compressReLU(auxiliary_in)
        to_input = cat([auxiliary_in.view(auxiliary.size(0), auxiliary.size(1), -1), emb], 2)
        output_list = []
        if separate == 1:
            for i in range(emb.size(0)):
                each_output, hidden = self.rnn(to_input[i,:,:].view(1,emb.size(1),-1), hidden)
                hidden = self.resetsent(hidden, input[i,:], eosidx)
                output_list.append(each_output)
            output = cat(output_list, 0)
        else:
            output, hidden = self.rnn(to_input, hidden)
        output = self.drop(output)

        if self.loss_type == 'nce':
            if self.mode == 'eval':
                decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)), mode='eval_full')
            else:
                decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)), target, mode='train', num_noise=self.num_noise)
            return decoded, hidden
        else:
            decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
            return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def resetsent(self, hidden, input, eosidx):
        if self.rnn_type == 'LSTM':
            outputcell = hidden[0]
            memorycell = hidden[1]
            mask = input != eosidx
            expandedmask = mask.unsqueeze(-1).expand_as(outputcell)
            expandedmask = expandedmask.float()
            return (outputcell*expandedmask, memorycell*expandedmask)
        else:
            mask = input != eosidx
            expandedmask = mask.unsqueeze(-1).expand_as(hidden)
            expandedmask = expandedmask.float()
            return hidden*expandedmask
