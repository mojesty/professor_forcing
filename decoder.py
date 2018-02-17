import torch
from torch import nn
from modules.attention import Attn
from torch.nn import functional as F


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embed_size, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(embed_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
        else:
            raise NotImplementedError

    def forward(self, word_inputs, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        seq_len, batch_size = word_inputs.size(1), word_inputs.size(0)

        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_inputs).view(seq_len, 1, -1)  # S=1 x B x N

        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))

        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights
