import random

import torch
from torch.autograd import Variable
from cfg import USE_CUDA, teacher_forcing_ratio, MAX_LENGTH, clip, max_length


class Trainer:
    def train(self, input, target, model, encoder_optimizer, decoder_optimizer, criterion):
        # save target length for decoding steps
        target_length = target.size(1)
        batch_size = input.size(0)

        # flatted target variable for loss
        #target_variable = target_variable.view(-1)

        # Zero gradients of both optimizers
        model.zero_grad()
        loss = 0

        # Run words through encoder
        decoder_outputs, _, _, _ = model(input, target, mode='training')
        for di in range(max_length):
            loss += criterion(decoder_outputs[di], target[:, di])

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm(model.decoder.parameters(), clip)
        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.data[0] / (target_length * batch_size)
