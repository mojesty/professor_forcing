import torch
from torch import nn
from torch.autograd import Variable

import cfg
from cfg import vocab, USE_CUDA
from modules.decoder import AttnDecoderRNN
from modules.encoder import EncoderRNN


class Translator(nn.Module):

    def __init__(
            self, vocab_size, embedding_size, hidden_size, n_layers, dropout_p, attn_model,
            encoder=None, decoder=None):
        super(Translator, self).__init__()

        # TODO: think how organize arguments
        self.encoder = EncoderRNN(
            vocab_size, embedding_size, hidden_size, n_layers
        ) if encoder is None else encoder
        self.decoder = AttnDecoderRNN(
            attn_model, embedding_size, hidden_size, cfg.model.vocab_size, n_layers, dropout_p=dropout_p
        ) if decoder is None else decoder

    def zero_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()

    def _sample(self, decoder_output, decoder_attention, context, method='multinomial', fill_unks='attn'):
        if method == 'argmax':
            topv, topi = decoder_output.data.topk(1)
        elif method == 'multinomial':
            # TODO: multi-inference for one sentence
            topi = torch.multinomial(decoder_output.cpu().exp().data, 1)
            if USE_CUDA: topi = topi.cuda()
        else:
            raise ValueError
        ni = topi[0][0]
        from_attn = False
        if ni == vocab.eos_idx:
            token_idx = vocab.eos_idx
        elif ni == vocab.unk_idx:
            if fill_unks is not None:
                # find the word in input sentence with the highest attention score and add it to the decoded words
                _, top_word_idx = decoder_attention.data.topk(1)
                # index_to_refer = input_variable.squeeze(0)[top_word_idx.view(-1)].cpu().data[0]
                # because maximum attention score can point to padding that is out of context, trim it
                index_to_refer = min(top_word_idx.view(-1)[0], len(context) - 1)
                token_idx = context[index_to_refer].cpu().data[0]
                from_attn = True
            else:
                token_idx = vocab.unk_idx
        else:
            token_idx = ni
        return token_idx, from_attn

    def forward(self, input, target=None, mode='inference', sample_method='multinomial', fill_unks='attn'):
        assert mode in ['training', 'inference']
        batch_size = input.size(0)
        encoder_hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = self.encoder(input, encoder_hidden)

        # start with <SOS> token for every sentence in minibatch
        decoder_input = Variable(torch.LongTensor([vocab.sos_idx]).repeat(1, batch_size)).t()
        # context of shape [batch_size x hidden_size]
        decoder_context = Variable(
            torch.zeros(batch_size, self.decoder.hidden_size * self.decoder.enc_num_directions)
        )
        decoder_hidden = encoder_hidden
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()

        indices = [vocab.sos_idx]
        from_attns = [False]
        decoder_attentions = torch.zeros(cfg.max_length, cfg.max_length)
        decoder_outputs = []

        for di in range(cfg.max_length):
            # TODO: return smth meaningful for training phase
            decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(
                decoder_input,
                decoder_context,
                decoder_hidden,
                encoder_outputs
            )
            if mode == 'training' and cfg.teacher_forcing:
                # Next target is next input
                decoder_input = target[:, di].unsqueeze(1)
                # TODO: possibly memory leaky code!
                decoder_outputs.append(decoder_output)

            elif mode == 'inference' or mode == 'training' and cfg.teacher_forcing:
                next_token_idx, from_attn = self._sample(
                    decoder_output,
                    decoder_attention,
                    input.squeeze(0),
                    method=sample_method,
                    fill_unks=fill_unks
                )
                if mode == 'inference':
                    # save additional data only for inference
                    indices.append(next_token_idx)
                    from_attns.append(from_attn)
                    decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

                decoder_input = Variable(torch.LongTensor([[next_token_idx]]))
                if USE_CUDA: decoder_input = decoder_input.cuda()
            else:
                raise AssertionError

            if (mode == 'training' and di == cfg.max_length - 1) or (mode == 'inference' and next_token_idx == vocab.eos_idx):
                break

        return decoder_outputs, indices, from_attns, decoder_attentions


