# TODO: organize it
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

# gonna format them later
ENC_DUMP_PATH = 'encoder_{}_bs128.binary'
DEC_DUMP_PATH = 'decoder_{}_bs128.binary'
LOGDIR = 'logs'  # for Tensorboard

LOSSDIR = 'losses.txt'  # for our purposes (o rly?)
n_epochs = 7

dropout_p = 0.2
vocab_size = 20000
n_layers = 1
hidden_size = 300
attn_model = 'general'

max_length = 30