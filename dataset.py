import torch
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable

from cfg import USE_CUDA, vocab, max_length, model


class QADataset(Dataset):
    """
    Simple QA Dataset. Refers to torchtext.vocab for tensor creating
    Max_length is the maximum length of its data (questiond and answer),
    instances are trimmed/padded if necessary
    n_special is number of special symbols
    Current special symbols:
    * sos
    * eos
    * unk
    See cfg.py for details
    """

    def __init__(self, data, vocab, max_length=max_length, gpu=True):
        self.vocab = vocab
        self.data = data
        self.max_length = max_length
        self.gpu = gpu
        self.n_special = 3

    def indexes_from_sentence(self, sentence):
        # be careful with it, as it preprocceses
        res = []
        for i, word in enumerate(sentence.split(' ')[:self.max_length]):
            if word == vocab.sos:
                res.append(0)
            elif word == vocab.eos:
                res.append(1)
            elif word in self.vocab.stoi and self.vocab.stoi[word] < model.vocab_size - 3:
                res.append(self.vocab.stoi[word] + 3)
            else:
                res.append(2)  # (self.vocab.stoi['unk'])
        # pad sequences for minibatch mode
        res = res + [vocab.eos_idx for _ in range(self.max_length - len(res))]
        return torch.cuda.LongTensor(res) if self.gpu else torch.LongTensor(res)

    def variable_from_sentence(self, sentence):
        indexes = self.indexes_from_sentence(sentence)
        # TODO: do we need varialbes?
        var = Variable(indexes)
        if USE_CUDA:
            var = var.cuda()
        return var

    def __getitem__(self, idx):
        return (
            self.indexes_from_sentence(self.data[idx]['context']),
            self.indexes_from_sentence(self.data[idx]['question']),
        )

    def __len__(self):
        return len(self.data)

    def itos(self, i):
        if i == 0:
            return vocab.sos
        elif i == 1:
            return vocab.eos
        elif i == 2:
            return vocab.unk
        else:
            return self.vocab.itos[i - 3]
