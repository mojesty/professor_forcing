import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from cfg import USE_CUDA

class Attn(nn.Module):
    def __init__(self, method, hidden_size, bidirectional):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            if not bidirectional:
                self.attn = nn.Linear(hidden_size, hidden_size)
            else:
                # TODO: assert
                self.attn = nn.Linear(hidden_size, hidden_size // 2)

        elif self.method == 'concat':
            raise NotImplementedError
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))
        else:
            raise NotImplementedError

    def forward(self, hidden, encoder_outputs):
        # hidden of shape          [batch_size x hidden_size]
        # encoder_outputs          [seq_len x batch_size x hidden_size]
        seq_len, batch_size = encoder_outputs.size(0), encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len, batch_size))
        # attn_energies:           [seq_len x batch_size]
        if USE_CUDA: attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):  # TODO: vectorize!
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies.t()).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            # vector version
            energy = torch.bmm(hidden.unsqueeze(1), energy.unsqueeze(2))
            # energy                 [batch_size x 1 x 1]
            return energy.squeeze()

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy