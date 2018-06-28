import torch
from torch import optim

import cfg


class Trainer:
    def __init__(self, opt, model):
        self.opt = opt
        self.model = model
        self.g_optim = optim.Adam(model.generator.parameters(), lr=opt.learning_rate)
        if opt.adversarial:
            self.d_optim = optim.Adam(model.discriminator.parameters(), lr=opt.learning_rate)

    def train(self, opt, input):

        self.model.zero_grad()
        nll_loss, tf_scores, ar_scores = self.model(input, adversarial=self.opt.adversarial)

        # Backpropagation
        nll_loss.backward(retain_graph=True)
        if self.opt.adversarial:
            g_loss = self._calculate_generator_loss(tf_scores, ar_scores).sum()
            d_loss = self._calcualte_discriminator_loss(tf_scores, ar_scores).sum()

            d_loss.backward(retain_graph=True)
            g_loss.backward()
            g_loss_value = g_loss.item()
            d_loss_value = d_loss.item()
        else:
            g_loss_value = None
            d_loss_value = None
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.clip)

        self.g_optim.step() and (not self.opt.adversarial or self.d_optim.step())

        return nll_loss.cpu().item(), g_loss_value, d_loss_value

    def _calculate_generator_loss(self, tf_scores, ar_scores):
        """
        Calculates Fool-The-Discriminator loss
        Optionally calculate the reverse loss
        :param tf_scores: Teacher Forcing scores
        :param ar_scores: AutoRegressive scores
        :return:
        """
        loss = torch.log(ar_scores) * (-1)

        if self.opt.optional_loss:
            loss += torch.log(1 - tf_scores) * (-1)
        return loss

    def _calcualte_discriminator_loss(self, tf_scores, ar_scores):
        tf_loss = torch.log(tf_scores) * (-1)
        ar_loss = torch.log(1 - ar_scores) * (-1)
        return tf_loss + ar_loss
