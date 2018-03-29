# TODO: organize it
import os

USE_CUDA = True

teacher_forcing_ratio = 1.0  # autoregressive training off
clip = 2.0
batch_size = 512

MAX_LENGTH = 30

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
ENC_DUMP_PATH = 'models/encoder_{}_2layers_nhid_1024.binary'
DEC_DUMP_PATH = 'models/decoder_{}_2layers_nhid_1024.binary'

# Tensorboard configs
# TODO: organize
LOGDIR = 'logs'
NAME = 'logs/300_10k_2l_bidir_adam'

LOSSDIR = 'losses.txt'  # for our purposes (o rly?)
n_epochs = 10


max_length = 30


class model:
    dropout_p = 0.2
    vocab_size = 5000
    n_layers = 1
    embedding_size = 300
    hidden_size = 400
    attn_model = 'general'

    bidirectional = False

class sample_methods:
    argmax = 'argmax'
    multinomial = 'multinomial'

teacher_forcing = True

class inits:
    xavier = 'xavied'
    zeros='zeros'
# for easy CUDA debug
# os.environ['CUDA_LAUNCH_BLOCKING'] = True
