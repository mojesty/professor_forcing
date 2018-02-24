import torch
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable

from cfg import USE_CUDA, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, EOS_TOKEN_IDX


class QADataset(Dataset):
    """
    Simple QA Dataset. Refers to torchtext.vocab for tensor creating
    """

    def __init__(self, data, vocab, max_length=30, gpu=True):
        self.vocab = vocab
        self.data = data
        self.max_length = max_length
        self.gpu = gpu

    def indexes_from_sentence(self, sentence):
        # be careful with it, as it preprocceses
        res = []
        for i, word in enumerate(sentence.split(' ')[:self.max_length]):
            if word == SOS_TOKEN:
                res.append(0)
            elif word == EOS_TOKEN:
                res.append(1)
            elif word in self.vocab.stoi and self.vocab.stoi[word] < 20000 - 3:
                res.append(self.vocab.stoi[word] + 3)
            else:
                res.append(2)  # (self.vocab.stoi['unk'])
        # pad sequences for minibatch mode
        res = res + [EOS_TOKEN_IDX for _ in range(self.max_length - len(res))]
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
            self.indexes_from_sentence(self.data[idx]['question'])
        )

    def __len__(self):
        return len(self.data)

    def itos(self, i):
        if i == 0:
            return SOS_TOKEN
        elif i == 1:
            return EOS_TOKEN
        elif i == 2:
            return UNK_TOKEN
        else:
            return self.vocab.itos[i - 3]
