import torch

USE_CUDA = True

device = 'cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu'


clip = 2.0
batch_size = 128


class vocab:
    sos = 'bos'
    eos = 'eos'
    unk = 'unk'
    sos_idx = 0
    eos_idx = 1
    unk_idx = 2

NEED_SAVE = True
NEED_LOAD = False
save = 'last'
assert save in ['all', 'last', 'best']  # TODO: last and best

# gonna format them later
ENC_DUMP_PATH = 'test_generator05.pt'
DEC_DUMP_PATH = 'models/decoder_{}_2layers_nhid_1024.binary'


LOGDIR = 'logs'
NAME = 'logs/300_10k_2l_bidir_adam'

LOSSDIR = 'losses.txt'  # for our purposes (o rly?)
n_epochs = 10
learning_rate = 0.0005

max_length = 100

data_instances = -1


class model:
    vocab_size = -1
    embedding_size = 32
    hidden_size = 1024


class sample_methods:
    argmax = 'argmax'
    multinomial = 'multinomial'


class inits:
    xavier = 'xavied'
    zeros = 'zeros'

