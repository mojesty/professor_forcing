# TODO: organize it
USE_CUDA = True

teacher_forcing_ratio = 1.0  # autoregressive training off
clip = 2.0
batch_size = 256

MAX_LENGTH = 30

SOS_TOKEN = 'bos'
EOS_TOKEN = 'eos'
UNK_TOKEN = 'unk'
SOS_TOKEN_IDX = 0
EOS_TOKEN_IDX = 1
UNK_TOKEN_IDX = 2

NEED_SAVE = True
NEED_LOAD = False

# gonna format them later
ENC_DUMP_PATH = 'encoder_{}_bs128_vocab40k.binary'
DEC_DUMP_PATH = 'decoder_{}_bs128_vocab40k.binary'
LOGDIR = 'logs'  # for Tensorboard

LOSSDIR = 'losses.txt'  # for our purposes (o rly?)
n_epochs = 7

dropout_p = 0.2
vocab_size = 30000
n_layers = 1
hidden_size = 300
attn_model = 'general'