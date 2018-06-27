from torch import nn

import cfg
from modules.discriminator import Discriminator
from modules.generator import Generator


class LMGan(nn.Module):

    def __init__(self, opt):
        super(LMGan, self).__init__()
        self.opt = opt
        self.generator = Generator(
            opt.vocab_size,
            opt.embedding_size,
            opt.hidden_size,
            opt.device
        )

        self.discriminator = Discriminator(
            opt.hidden_size,
            opt.d_hidden_size,
            opt.d_linear_size,
            opt.d_dropout,
            opt.device
        )

    def forward(self, input, adversarial=True):
        batch_size = input.size(0)
        if not adversarial:
            # vanilla Negative log-likelihood training, no sampling
            start_hidden = self.generator.init_hidden(batch_size, strategy=cfg.inits.zeros)

            loss, gen_hidden_states, _ = self.generator.consume(input, start_hidden, sampling=False)

            return loss, None, None
        else:
            # run one pass without sampling
            start_hidden_nll = self.generator.init_hidden(batch_size, strategy=cfg.inits.zeros)
            loss_nll, gen_hidden_states_nll, _ = self.generator.consume(
                input, start_hidden_nll, sampling=False)

            # run one pass with sampling
            start_hidden_adv = self.generator.init_hidden(batch_size, strategy=cfg.inits.zeros)
            loss_adv, gen_hidden_states_adv, _ = self.generator.consume(
                input,
                start_hidden_adv,
                sampling=True,
                method=self.opt.sampling_strategy,
                temperature=self.opt.temperature
            )
            # these two passes have computational graphs that are completely different, so
            # in the future we can call backwards for each loss consequently

            # Now, call the discriminator
            teacher_forcing_scores = self.discriminator(gen_hidden_states_nll)
            autoregressive_scores = self.discriminator(gen_hidden_states_adv)

            return loss_nll + loss_adv, teacher_forcing_scores, autoregressive_scores

