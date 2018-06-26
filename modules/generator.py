import cfg

import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F

multinomial = cfg.sample_methods.multinomial

class Generator(nn.Module):
    def __init__(
            self, vocab_size, embedding_size, hidden_size
    ):
        # input size means embedding size as usual
        super(Generator, self).__init__()

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

        # scores = F.softmax(unnormalized_scores, dim=1)    # [batch_size x vocab_size]
        return unnormalized_scores, next_hidden

    def consume(self, word_input, hidden, sampling, method=multinomial):
        # word_inputs                                       [batch_size x seq_len]
        # hidden                                            [batch_size x hidden_size]
        # store all hidden states for discriminator
        hidden_states = [hidden]
        word_inputs = [word_input.data]
        seq_len, batch_size = word_input.size(1), word_input.size(0)
        criterion = nn.CrossEntropyLoss()
        loss = 0
        if sampling:
            # autoregressive mode
            # we need only initial word inputs
            current_word_inputs = word_input[:, 1].unsqueeze(1)
            for idx in range(seq_len - 1):
                scores, hidden = self(current_word_inputs, hidden)
                loss += criterion(scores, word_input[:, idx + 1])
                hidden_states.append(hidden)
                current_word_inputs = self._sample(scores, method)
                word_inputs.append(current_word_inputs.data)

        else:
            # teacher forcing mode
            for idx in range(seq_len - 1):
                scores, hidden = self(word_input[:, idx].unsqueeze(1), hidden)
                loss += criterion(scores, word_input[:, idx + 1])
                hidden_states.append(hidden)


        # we still can't go backward because we another losses are not computed
        return loss, hidden_states, word_inputs

    def _sample(self, scores, method):
        # scores                                            [batch_size x vocab_size]
        # method                                            cfg.sample_methods
        scores = F.softmax(scores, dim=1)
        if method == 'argmax':
            # find elements with max values for each score list in the minibatch
            topv, topi = scores.data.topk(1, dim=1)       # [batch_size x 1]
        elif method == 'multinomial':
            _scores = scores.cpu().exp().data
            topi = torch.multinomial(_scores, 1)          # [batch_size x 1]
        topi = topi.to(cfg.device)
        return topi

    def init_hidden(self, batch_size, strategy=cfg.inits.xavier):
        if strategy == cfg.inits.zeros:
            hidden = torch.zeros(batch_size, self.hidden_size)
        elif strategy == cfg.inits.xavier:
            hidden = torch.zeros(batch_size, self.hidden_size)
            hidden = Variable(torch.nn.init.xavier_normal(hidden))
        hidden = hidden.to(cfg.device)
        return hidden


if __name__ == '__main__':
    encoder_test = Generator(
        vocab_size=31, hidden_size=10)
    print(encoder_test)

    encoder_hidden = encoder_test.init_hidden(batch_size=1).cpu()
    # word_input = torch.LongTensor([1, 2, 3]).view(1, -1).to(cfg.device)
    # if USE_CUDA:
    #     encoder_test.cuda()
    #     word_input = word_input.cuda()
    encoder_outputs, encoder_hidden = encoder_test(word_input, encoder_hidden)
