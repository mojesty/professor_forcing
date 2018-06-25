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

        next_hidden = self.gru(embedded, hidden)          # [batch_size x hidden_size]

        unnormalized_scores = self.linear(next_hidden)    # [batch_size x vocab_size]

        scores = F.softmax(unnormalized_scores, dim=1)    # [batch_size x vocab_size]
        return scores

    def consume(self, word_inputs, start_hidden, sampling, method=multinomial):
        # word_inputs                                       [batch_size x seq_len]
        # hidden                                            [batch_size x hidden_size]
        if sampling:
            # autoregressive mode



    def _sample(self, scores, method):
        # scores                                            [batch_size x vocab_size]
        if method == 'argmax':
            topv, topi = scores.data.topk(1)
        elif method == 'multinomial':
            topi = torch.multinomial(scores.cpu().exp().data, 1)
        topi.to(cfg.device)
        return topi

    def init_hidden(self, batch_size, strategy=cfg.inits.xavier):
        if strategy == cfg.inits.zeros:
            hidden = Variable(torch.zeros(
                self.n_layers * self.num_directions, batch_size, self.hidden_size))
        elif strategy == cfg.inits.xavier:
            hidden = torch.zeros(
                self.n_layers * self.num_directions, batch_size, self.hidden_size)
            hidden = Variable(torch.nn.init.xavier_normal(hidden))
        hidden = hidden.to(cfg.device)
        return hidden


if __name__ == '__main__':
    encoder_test = Generator(
        vocab_size=31, hidden_size=10)
    print(encoder_test)

    encoder_hidden = encoder_test.init_hidden(batch_size=1).cpu()
    word_input = torch.LongTensor([1, 2, 3]).view(1, -1).to(cfg.device)
    # if USE_CUDA:
    #     encoder_test.cuda()
    #     word_input = word_input.cuda()
    encoder_outputs, encoder_hidden = encoder_test(word_input, encoder_hidden)
