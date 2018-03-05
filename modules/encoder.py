import unicodedata
import string
import re
import random
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from cfg import USE_CUDA


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        # input size means embedding size as usual
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size,
                                      hidden_size)  # input_size means vocab size, hidden_size means embedding_dim
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)  # for simplicity

    def forward(self, word_inputs, hidden):
        # word_inputs of shape (batch_size * seq_len)
        # hidden of shape (seq_len * batch_size * hidden_size)
        seq_len, batch_size = word_inputs.size(1), word_inputs.size(0)
        # embedding layer requires word_input of shape (N, W)
        embedded = self.embedding(word_inputs).view(seq_len, batch_size, -1)
        # embedded of shape (seq_len * batch_size * hidden_size)
        output, hidden = self.gru(embedded, hidden)
        # output of shape (seq_len * batch_size * hidden_size)
        # hidden does not change its shape
        return output, hidden

    def init_hidden(self, batch_size):
        # TODO: different initialization strategies
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if USE_CUDA:
            hidden = hidden.cuda()
        return hidden


if __name__ == '__main__':
    encoder_test = EncoderRNN(
        input_size=31, hidden_size=10, n_layers=1)
    print(encoder_test)

    encoder_hidden = encoder_test.init_hidden().cpu()
    word_input = Variable(torch.LongTensor([1, 2, 3]).view(1, -1))  # (N, length) where N is mini-batch size
    # if USE_CUDA:
    #     encoder_test.cuda()
    #     word_input = word_input.cuda()
    encoder_outputs, encoder_hidden = encoder_test(word_input, encoder_hidden)
