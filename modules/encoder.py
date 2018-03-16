from cfg import model

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from cfg import USE_CUDA


class EncoderRNN(nn.Module):
    def __init__(
            self, vocab_size, embedding_size, hidden_size, n_layers=1, dropout_p=0.2,
            bidirectional=model.bidirectional
    ):
        # input size means embedding size as usual
        super(EncoderRNN, self).__init__()

        self.input_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )  # input_size means vocab size, hidden_size means embedding_dim
        self.gru = nn.GRU(
            embedding_size,
            hidden_size,
            n_layers,
            dropout=dropout_p,
            bidirectional=bidirectional
        )

    def forward(self, word_inputs, hidden):
        # word_inputs of shape [batch_size x seq_len]
        # hidden [seq_len x batch_size x hidden_size * num_directions]

        seq_len, batch_size = word_inputs.size(1), word_inputs.size(0)

        # embedding layer requires word_input of shape [N x W]
        embedded = self.embedding(word_inputs).view(seq_len, batch_size, -1)
        # [seq_len x batch_size x embedding_size]
        output, hidden = self.gru(embedded, hidden)
        # output [seq_len x batch_size x hidden_size * num_directions]

        # we save only those hidden state that corresponds to forward pass
        hidden = hidden.view(self.n_layers, -1, batch_size, self.hidden_size)[:, 0, :, :].contiguous()
        # hidden [n_layers x batch_size x hidden_size]
        return output, hidden

    def init_hidden(self, batch_size):
        # TODO: different initialization strategies
        hidden = Variable(torch.zeros(
            self.n_layers * self.num_directions, batch_size, self.hidden_size))
        if USE_CUDA:
            hidden = hidden.cuda()
        return hidden


if __name__ == '__main__':
    encoder_test = EncoderRNN(
        vocab_size=31, hidden_size=10, n_layers=1)
    print(encoder_test)

    encoder_hidden = encoder_test.init_hidden().cpu()
    word_input = Variable(torch.LongTensor([1, 2, 3]).view(1, -1))  # (N, length) where N is mini-batch size
    # if USE_CUDA:
    #     encoder_test.cuda()
    #     word_input = word_input.cuda()
    encoder_outputs, encoder_hidden = encoder_test(word_input, encoder_hidden)
