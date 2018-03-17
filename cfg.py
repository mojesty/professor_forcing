# TODO: organize it
import os

USE_CUDA = True

teacher_forcing_ratio = 1.0  # autoregressive training off
clip = 2.0
batch_size = 256

MAX_LENGTH = 30

class vocab:

    sos = 'bos'
    eos = 'eos'
    unk = 'unk'
    sos_idx = 0
    eos_idx = 1
    unk_idx = 2

NEED_SAVE = False
NEED_LOAD = False
save = 'last'
assert save in ['all', 'last', 'best']  # TODO: last and best

# gonna format them later
ENC_DUMP_PATH = 'models/encoder_{}_2layers_bidir.binary'
DEC_DUMP_PATH = 'models/decoder_{}_2layers_bidir.binary'

# Tensorboard configs
# TODO: organize
LOGDIR = 'logs'
NAME = 'logs/300_40k_2l_bidir_adam'

LOSSDIR = 'losses.txt'  # for our purposes (o rly?)
n_epochs = 18


max_length = 30


class model:
    dropout_p = 0.2
    vocab_size = 40000
    n_layers = 2
    embedding_size = 300
    hidden_size = 300
    attn_model = 'general'

    bidirectional = True

class sample_methods:
    argmax = 'argmax'
    multinomial = 'multinomial'

# for easy CUDA debug
# os.environ['CUDA_LAUNCH_BLOCKING'] = True
