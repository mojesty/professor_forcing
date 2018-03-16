import torch
from torch import nn
from modules.attention import Attn
from torch.nn import functional as F

from cfg import USE_CUDA, model


class AttnDecoderRNN(nn.Module):
    def __init__(
            self, attn_model, embedding_size, hidden_size, output_size,
            n_layers=1, dropout_p=0.1, bidirectional=model.bidirectional):
        super(AttnDecoderRNN, self).__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.enc_num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size

        # Define layers
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(
            hidden_size * self.enc_num_directions + embedding_size,
            hidden_size, n_layers, dropout=dropout_p
        )
        self.out = nn.Linear(
            hidden_size * self.enc_num_directions + hidden_size, output_size
        )

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(
                attn_model, hidden_size * self.enc_num_directions, bidirectional
            )

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # word_input:           [1 x batch_size x ???[
        # last_context:         [batch_size x hidden_size]
        # last_hidden:          [1 x batch_size x hidden_size]
        # encoder_outputs:      [seq_len x batch_size x hidden_size]

        batch_size = word_input.size(0)
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, batch_size, -1)
        # word_embedded:        [1 x batch_size x embedding_size]

        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        # rnn_input:            [1 x batch_size x (embedding_size + hidden_size * dirs)]
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        # attn_weights:         [batch_size x 1 x seq_len]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # context:              [batch_size x seq_len x hidden_size]

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0)
        # rnn_output:           [batch_size x hidden_size]
        context = context.squeeze(1)
        # context:              [batch_size x hidden_size]
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        # output:               [batch_size x output_size]
        # Return final output, context, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights
