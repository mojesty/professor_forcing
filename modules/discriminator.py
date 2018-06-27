import cfg

import torch
import torch.nn as nn

import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, linear_size, lin_dropout, device):
        super(Discriminator, self).__init__()

        self.hidden_size = hidden_size
        self.linear_size = linear_size
        self.device = device

        self.rnn = nn.GRU(
            input_size, hidden_size,
            batch_first=True, bidirectional=True
        )

        self.linears = nn.Sequential(
            nn.Linear(hidden_size * 2, linear_size),
            nn.ReLU(),
            nn.Dropout(lin_dropout),
            nn.Linear(linear_size, linear_size),
            nn.ReLU(),
            nn.Dropout(lin_dropout),
            nn.Linear(linear_size, 1)
        )

    def forward(self, hidden_states):
        # hidden_states                                      # [batch_size * seq_len * hid_size]
        initial_hidden = self.init_hidden(hidden_states.size(0))
        _, rnn_final_hidden = self.rnn(
            hidden_states, initial_hidden)                   # [batch_size * hid_size * 2]
        unnormalized_scores = self.linears(rnn_final_hidden) # [batch_size * 1]
        scores = F.softmax(unnormalized_scores, dim=1)       # [batch_size * 1]
        return scores

    def init_hidden(self, batch_size):
        with torch.no_grad:
            hidden = torch.zeros(batch_size, self.hidden_size * 2)
        hidden = hidden.to(self.device)
        return hidden