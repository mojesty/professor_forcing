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

USE_CUDA = False


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
        seq_len, batch_size = word_inputs.size(1), word_inputs.size(0)
        # embedding layer requires word_input of shape (N, W)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden


encoder_test = EncoderRNN(
    input_size=31, hidden_size=10, n_layers=1)
print(encoder_test)

encoder_hidden = encoder_test.init_hidden().cpu()
word_input = Variable(torch.LongTensor([1, 2, 3]).view(1, -1))  # (N, length) where N is mini-batch size
# if USE_CUDA:
#     encoder_test.cuda()
#     word_input = word_input.cuda()
encoder_outputs, encoder_hidden = encoder_test(word_input, encoder_hidden)
