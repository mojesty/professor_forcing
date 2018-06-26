import torch
from torch.utils.data.dataset import Dataset

from cfg import device, data_instances


class LMDataset(Dataset):
    """
    Simple LM Dataset. Refers to vocab for indexing.
    """

    def __init__(self, data, vocab):
        self.vocab = vocab
        self.data = data

    def indexes_from_sentence(self, sentence):
        res = [self.vocab.stoi(char) for char in sentence.split(' ') if char]
        return torch.LongTensor(res).to(device)

    def __getitem__(self, idx):
        return self.indexes_from_sentence(self.data[idx])

    def __len__(self):
        return len(self.data) if data_instances < 0 else data_instances


class Vocab:
    d = {' ': 4, '#': 14, '$': 13, '&': 45, "'": 23, '*': 31, '-': 30,
         '.': 1,
         '/': 28,
         '0': 7,
         '1': 47,
         '2': 39,
         '3': 48,
         '4': 34,
         '5': 11,
         '6': 3,
         '7': 38,
         '8': 33,
         '9': 40,
         '<': 10,
         '>': 21,
         'N': 36,
         '\\': 25,
         '_': 15,
         'a': 20,
         'b': 49,
         'c': 29,
         'd': 32,
         'e': 46,
         'f': 5,
         'g': 16,
         'h': 22,
         'i': 0,
         'j': 41,
         'k': 2,
         'l': 43,
         'm': 26,
         'n': 35,
         'o': 37,
         'p': 42,
         'q': 6,
         'r': 18,
         's': 24,
         't': 9, 'u': 12, 'v': 19, 'w': 8, 'x': 17, 'y': 44, 'z': 27}
    d_ = dict({v: k for k, v in d.items()})

    def itos(self, idx):
        return self.d_[idx]

    def stoi(self, char):
        return self.d[char]


