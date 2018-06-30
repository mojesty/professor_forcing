import pickle

import torch
from torch.utils.data.dataset import Dataset

from cfg import data_instances, vocab

unk = vocab.unk


class LMDataset(Dataset):
    """
    Simple LM Dataset. Refers to vocab for indexing.
    """

    def __init__(self, corpus_path, vocab_path,
                 bptt, device, min_counts):
        self.seq_lengths = bptt
        self.device = device
        # set up vocab
        self.vocab = Vocab(corpus_path, min_counts) if not vocab_path\
            else pickle.load(open(vocab_path, 'rb'))
        self.vocab.min_counts = min_counts
        if not vocab_path:
            pickle.dump(self.vocab, open('vocab.pt', 'wb'))

        _data = []
        with open(corpus_path, 'r') as f:
            for line in f:
                _data.append(line[:-1])  # remove possible \n
            f.close()
        _data = ''.join(_data)
        # truncate data to make even splits
        _data = _data[:(len(_data) // bptt) * bptt]
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

    def __init__(self, corpus_path, min_counts):
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
        self.vocab = {unk: 0}
        for k, v in self.counts.items():
            self.vocab[k] = len(self.vocab)
        self.vocab_ = {v: k for k, v in self.vocab.items()}
        self.min_counts = min_counts
        print('Vocabulary built')

    def itos(self, idx):
        cand = self.vocab_[idx]
        if self.counts[cand] >= self.min_counts:
            return cand
        else:
            return unk

    def stoi(self, token):
        if self.counts[token] >= self.min_counts:
            return self.vocab[token]
        else:
            return self.vocab[unk]

    def __getitem__(self, token):
        # TODO: fix it
        return self.stoi(token)

    def __len__(self):
        return len(
            [k for k, v in self.counts.items() if v >= self.min_counts]
        )


