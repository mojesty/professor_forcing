import random

import torch
from torch.autograd import Variable
from cfg import USE_CUDA, teacher_forcing_ratio, MAX_LENGTH, EOS_TOKEN_IDX, SOS_TOKEN_IDX, clip


class Trainer:
    def train(self, input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
        # save target length for decoding steps
        target_length = target_variable.size(1)
        batch_size = input_variable.size(0)

        # flatted target variable for loss
        #target_variable = target_variable.view(-1)

        # Zero gradients of both optimizers
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss = 0  # Added onto for each word

        # initialize hidden state wrt batch size
        encoder_hidden = encoder.init_hidden(batch_size)
        # Run words through encoder
        encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

        # input of shape batch_size * 1
        decoder_input = Variable(torch.LongTensor([SOS_TOKEN_IDX]).repeat(1, batch_size)).t()
        # context of shape batch_size * hidden_size
        decoder_context = Variable(torch.zeros(batch_size, decoder.hidden_size))
        # Use last hidden state from encoder to start decoder
        decoder_hidden = encoder_hidden
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
                loss += criterion(decoder_output, target_variable[:, di])
                decoder_input = target_variable[:, di]  # Next target is next input

        else:
            raise NotImplementedError
            # Without teacher forcing: use network's own prediction as the next input
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
                    decoder_input,
                    decoder_context,
                    decoder_hidden,
                    encoder_outputs
                )
                loss += criterion(decoder_output, target_variable[:, di])  # decoder_output[0]

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

        return loss.data[0] / (target_length * batch_size)