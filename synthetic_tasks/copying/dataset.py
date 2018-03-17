from torch.utils.data.dataset import Dataset
import torch
from synthetic_tasks.copying.cfg import dataset, n_epochs, gpu

import numpy as np


class CopyDataset(Dataset):

    def __init__(self, vocab_size, num_chars, num_pad):
        assert vocab_size > 1  # otherwise self.copying_data will fail
        self.vocab_size = vocab_size
        self.num_chars = num_chars
        self.num_pad = num_pad

    def copying_data(self):
        seq = np.random.randint(1, high=self.vocab_size, size=(1, self.num_chars))
        zeros1 = np.zeros((1, self.num_pad - 1))
        zeros2 = np.zeros((1, self.num_pad))
        marker = (self.vocab_size - 1) * np.ones((1, 1))
        zeros3 = np.zeros((1, 5))

        x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int64')
        y = np.concatenate((zeros3, zeros2, seq), axis=1).astype('int64')
        source = self.to_gpu(torch.from_numpy(x).squeeze(0).type(torch.LongTensor))
        target = self.to_gpu(torch.from_numpy(y).squeeze(0).type(torch.LongTensor))
        return source, target

    def to_gpu(self, inp):
        if gpu:
            return inp.cuda()
        else:
            return inp.cpu()

    def __getitem__(self, idx):
        return self.copying_data()

    def __len__(self):
        return n_epochs
