import torch.nn as nn

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def rnn_module_name(self, i):
        return 'rnn_{}'.format(i)

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            # self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            for layer in range(nlayers):
                self.add_module(self.rnn_module_name(layer), getattr(nn, rnn_type)(ninp, nhid, 1, dropout=dropout))
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
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

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        output = self.drop(self.encoder(input))
        # output, hidden = self.__dict__[rnn_module_name(layer)](output, hidden)
        # output = self.drop(output)
        for layer in range(self.nlayers):
            output, hidden[layer] = getattr(self, self.rnn_module_name(layer))(output, hidden[layer])
            output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def flatten_parameters(self):
        for layer in range(self.nlayers):
            getattr(self, self.rnn_module_name(layer)).flatten_parameters()

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            # return (weight.new_zeros(1, bsz, self.nhid),
            #         weight.new_zeros(1, bsz, self.nhid))
            return [(weight.new_zeros(1, bsz, self.nhid),
                    weight.new_zeros(1, bsz, self.nhid)) for layers in range(self.nlayers)]
        else:
            return [weight.new_zeros(1, bsz, self.nhid) for layers in range(self.nlayers)]
