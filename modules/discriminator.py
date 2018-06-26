import cfg

import torch
import torch.nn as nn

import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, linear_size):
        super(Discriminator, self).__init__()

        self.hidden_size = hidden_size
        self.linear_size = linear_size

        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)

        self.linear = nn.Linear(hidden_size, linear_size)

    def forward(self, hidden_states):

        rnn_output, _ = self.rnn(hidden_states)

