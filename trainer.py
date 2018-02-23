import random

import torch
from torch.autograd import Variable
from cfg import USE_CUDA, teacher_forcing_ratio, MAX_LENGTH, EOS_TOKEN_IDX, SOS_TOKEN_IDX, clip


class Trainer:
    def train(self, input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
              max_length=MAX_LENGTH):

        # input_variable = input_variable.view(-1)
        target_variable = target_variable.view(-1)

        # Zero gradients of both optimizers
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss = 0  # Added onto for each word

        # Get size of input and target sentences
        input_length = input_variable.size(1)
        target_length = target_variable.size(0)

        # Run words through encoder
        encoder_hidden = encoder.init_hidden()
        encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([[SOS_TOKEN_IDX]]))
        decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
        decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()

        # Choose whether to use teacher forcing
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        if use_teacher_forcing:

            # Teacher forcing: Use the ground-truth target as the next input
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
                    decoder_input,
                    decoder_context,
                    decoder_hidden,
                    encoder_outputs
                )
                loss += criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]  # Next target is next input

        else:
            # Without teacher forcing: use network's own prediction as the next input
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                                             decoder_context,
                                                                                             decoder_hidden,
                                                                                             encoder_outputs)
                loss += criterion(decoder_output, target_variable[di])  # decoder_output[0]

                # Get most likely word index (highest value) from output
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))  # Chosen word is next input
                if USE_CUDA: decoder_input = decoder_input.cuda()

                # Stop at end of sentence (not necessary when using known targets)
                if ni == EOS_TOKEN_IDX: break

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.data[0] / target_length