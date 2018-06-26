import pickle

import torch

import cfg
from dataset import LMDataset, Vocab

multinomial = cfg.sample_methods.multinomial


class Sampler:

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def sample(self, batch_size=1, input=None, start_hidden=None, strategy=multinomial):
        """
        Main sampling function.
        :param batch_size:
        :param start_hidden:
        :return:
        """
        self.model.eval()
        start_hidden = start_hidden or self.model.init_hidden(batch_size)

        # run model and get hidden states and sampled indices
        input = input or self.input(batch_size)
        _, hidden_states, words_indices = self.model.consume(
            input, start_hidden, sampling=True, method=strategy)

        words_indices = torch.stack(words_indices[1:], dim=1).cpu()

        for idx in range(batch_size):
            tokens = [self.dataset.vocab.itos(idx[0].item()) for idx in words_indices[idx, :]]
            print(''.join(tokens))

    def input(self, batch_size):
        l = [0] * len(self.dataset.vocab.d)
        l[10] = 1
        return torch.LongTensor(l).to(cfg.device).repeat((batch_size, 1))

if __name__ == '__main__':
    model = torch.load(cfg.ENC_DUMP_PATH)
    model.to(cfg.device)

    vocab = pickle.load(open('vocab.pt', 'rb'))
    corpus = pickle.load(open('data.pt', 'rb'))
    lmdataset = LMDataset(vocab=vocab, data=corpus)
    sampler = Sampler(model, lmdataset)

    sampler.sample(batch_size=1)