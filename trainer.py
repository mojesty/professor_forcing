import torch


import cfg


class Trainer:
    def train(self, input, model, generator_optimizer, decoder_optimizer, criterion):
        batch_size = input.size(0)

        # Zero gradients of both optimizers
        start_hidden = model.init_hidden(batch_size, strategy=cfg.inits.zeros)
        model.zero_grad()

        # Run words through encoder
        loss, _, _ = model.consume(input, start_hidden, sampling=False)

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), cfg.clip)
        generator_optimizer.step()
        # decoder_optimizer.step()

        return loss.cpu().item()
