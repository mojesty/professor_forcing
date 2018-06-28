import cfg

import torch
import torch.nn as nn

import torch.nn.functional as F

multinomial = cfg.sample_methods.multinomial


class Generator(nn.Module):
    def __init__(
            self, vocab_size, embedding_size, hidden_size, device=None
    ):
        # input size means embedding size as usual
        super(Generator, self).__init__()

        self.device = device
        self.input_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )  # input_size means vocab size, hidden_size means embedding_dim

        self.rnn = nn.GRUCell(
            embedding_size,
            hidden_size
        )

        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, word_inputs, hidden):
        # word_inputs                                       [batch_size x 1]
        # hidden                                            [batch_size x hidden_size]

        seq_len, batch_size = word_inputs.size(1), word_inputs.size(0)

        # embedding layer requires word_input of shape [N x W]
        embedded = self.embedding(
            word_inputs).view(batch_size, -1)             # [batch_size x emb_size]

        next_hidden = self.rnn(embedded, hidden)          # [batch_size x hidden_size]

        unnormalized_scores = self.linear(next_hidden)    # [batch_size x vocab_size]

        # scores = F.softmax(unnormalized_scores, dim=1)  # [batch_size x vocab_size]
        return unnormalized_scores, next_hidden

    def consume(self, word_input, hidden, sampling,
                method=multinomial, temperature=3, n_sampled=None):
        # word_inputs                                       [batch_size x seq_len]
        # hidden                                            [batch_size x hidden_size]
        # store all hidden states for discriminator
        hidden_states = [hidden]
        word_inputs = [word_input.data]
        seq_len, batch_size = word_input.size(1), word_input.size(0)
        if n_sampled: seq_len = n_sampled
        criterion = nn.CrossEntropyLoss()
        loss = 0
        if sampling:
            # autoregressive mode
            # we need only initial word inputs
            current_word_inputs = word_input[:, 1].unsqueeze(1)
            for idx in range(seq_len - 1):
                scores, hidden = self(current_word_inputs, hidden)
                if not n_sampled: loss += criterion(scores, word_input[:, idx + 1])
                hidden_states.append(hidden)
                current_word_inputs = self._sample(scores, method, temperature)
                word_inputs.append(current_word_inputs.data)

        else:
            # teacher forcing mode
            for idx in range(seq_len - 1):
                scores, hidden = self(word_input[:, idx].unsqueeze(1), hidden)
                loss += criterion(scores, word_input[:, idx + 1])
                hidden_states.append(hidden)

        # we still can't go backward because we another losses are not computed
        hidden_states = torch.stack(hidden_states, dim=1)
        # word_inputs = torch.stack(word_inputs, dim=1)
        return loss, hidden_states, word_inputs

    def _sample(self, scores, method, temperature):
        # scores                                            [batch_size x vocab_size]
        # method                                            cfg.sample_methods
        scores = F.softmax(scores, dim=1)
        if method == 'argmax':
            # find elements with max values for each score list in the minibatch
            topv, topi = scores.data.topk(1, dim=1)       # [batch_size x 1]
        elif method == 'multinomial':
            _scores = scores.cpu().div(temperature).exp().data
            topi = torch.multinomial(_scores, 1)          # [batch_size x 1]
        topi = topi.to(self.device)
        return topi

    def init_hidden(self, batch_size, strategy=cfg.inits.xavier):
        if strategy == cfg.inits.zeros:
            hidden = torch.zeros(batch_size, self.hidden_size)
        elif strategy == cfg.inits.xavier:
            hidden = torch.zeros(batch_size, self.hidden_size)
            hidden = torch.nn.init.xavier_normal_(hidden)
        hidden = hidden.to(self.device)
        return hidden

