import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from synthetic_tasks.copying.cfg import gpu


class CopyNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        # В нашем случае vocab_size -- это число разных букв во входной последовательности
        super(CopyNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.GRU(embedding_dim, hidden_dim)
        self.hidden2char = nn.Linear(hidden_dim, vocab_size)

        # make embeddings one-hot matrix for simplicity
        matrix = torch.eye(embedding_dim)
        self.embeddings._parameters['weight'] = torch.nn.parameter.Parameter(
            matrix, requires_grad=False
        )

    def init_hidden(self, batch_size):
        if gpu:
            return Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
        else:
            return Variable(torch.zeros(1, batch_size, self.hidden_dim))

    def forward(self, input):
        seq_len, batch_size = input.size(1), input.size(0)
        embeds = self.embeddings(input).view(seq_len, batch_size, -1)
        output, _ = self.lstm(embeds, self.hidden)

        char_space = self.hidden2char(output)
        char_scores = F.log_softmax(char_space).view(-1, self.vocab_size)  # for NLLLoss
        return char_scores
