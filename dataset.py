import pickle

import torch
from torch.utils.data.dataset import Dataset

from cfg import data_instances


class LMDataset(Dataset):
    """
    Simple LM Dataset. Refers to vocab for indexing.
    """

    def __init__(self, corpus_path, vocab_path, bptt, device):
        self.seq_lengths = bptt
        self.device = device
        self.vocab = Vocab(corpus_path) if not vocab_path\
            else pickle.load(open(vocab_path, 'rb'))
        if not vocab_path:
            pickle.dump(self.vocab, open('vocab.pt', 'wb'))
        # TODO: laze dataset
        _data = []
        with open(corpus_path, 'r') as f:
            for line in f:
                _data.append(line[:-1])  # remove possible \n
            f.close()
        _data = ''.join(_data)
        # split data with chunks of lengths bptt to use them further
        # in the __getitem__
        self.data = [_data[i:i + bptt]
            for i in range(0, len(_data), bptt)
        ]

    def indexes_from_sentence(self, sentence):
        res = [self.vocab[token] for token in sentence.split(' ') if token]
        return torch.LongTensor(res).to(self.device)

    def __getitem__(self, idx):
        return self.indexes_from_sentence(self.data[idx])

    def __len__(self):
        return len(self.data) if data_instances < 0 else data_instances


class Vocab:

    def __init__(self, corpus_path, min_counts=0):
        # online counting for memory efficiency
        self.counts = {}
        print('Building vocabulary...')
        with open(corpus_path, 'r') as f:
            for line in f:
                for token in line.split():
                    prev_counts = self.counts.get(token, 0)
                    self.counts[token] = prev_counts + 1
            f.close()

        # build vocabularies
        self.vocab = {}
        _c = 0
        for k, v in self.counts.items():
            if v >= min_counts:
                self.vocab[k] = _c
                _c += 1
        self.vocab_ = {v: k for k, v in self.vocab.items()}
        print('Vocabulary built')

    def itos(self, idx):
        return self.vocab_[idx]

    def stoi(self, char):
        return self.vocab[char]

    def __getitem__(self, token):
        # TODO: fix it
        return self.vocab.get(token, 'UNK')

    def __len__(self):
        return len(self.vocab)


