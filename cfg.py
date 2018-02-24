USE_CUDA = True

teacher_forcing_ratio = 1.0  # autoregressive training off
clip = 5.0
batch_size = 512

MAX_LENGTH = 30

SOS_TOKEN = 'bos'
EOS_TOKEN = 'eos'
UNK_TOKEN = 'unk'
SOS_TOKEN_IDX = 0
EOS_TOKEN_IDX = 1
UNK_TOKEN_IDX = 2

NEED_SAVE = False
NEED_LOAD = False

ENC_DUMP_PATH = 'encoder2.binary'
DEC_DUMP_PATH = 'decoder2.binary'

