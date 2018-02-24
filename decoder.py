import torch
from torch import nn
from modules.attention import Attn
from torch.nn import functional as F

from cfg import USE_CUDA


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # word_input of shape (1, batch_size, ???)
        # last_context of shape (batch_size, hidden_size)
        # last_hidden of shape (1, batch_size, hidden_size)
        # encoder_outputs of shape (seq_len, batch)size, hidden_size)

        batch_size = word_input.size(0)
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, batch_size, -1)
        # word_embedded of shape (1 * batch_size * embedding_size)

        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        # rnn_input of shape (1 x batch_size x (embedding_size * 2))
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        # attn_weights of shape (batch_size, 1, seq_len)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # context of shape (batch_size, seq_len, hidden_size)

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0)
        # rnn_output of shape (batch_size, hidden_size)
        context = context.squeeze(1)
        # context of shape (batch_size, hidden_size)
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        # output of shape (batch_size, output_size)
        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights
