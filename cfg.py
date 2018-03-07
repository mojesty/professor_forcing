# TODO: organize it
USE_CUDA = True

teacher_forcing_ratio = 1.0  # autoregressive training off
clip = 2.0
batch_size = 64

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

# gonna format them later
ENC_DUMP_PATH = 'models/encoder_{}_2layers.binary'
DEC_DUMP_PATH = 'models/decoder_{}_2layers.binary'
LOGDIR = 'logs'  # for Tensorboard

LOSSDIR = 'losses.txt'  # for our purposes (o rly?)
n_epochs = 30

dropout_p = 0.2
vocab_size = 40000
n_layers = 2
hidden_size = 300
attn_model = 'general'

max_length = 30