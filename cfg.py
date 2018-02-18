USE_CUDA = True

teacher_forcing_ratio = 0.5
clip = 5.0
MAX_LENGTH = 30

SOS_TOKEN = 'bos'
EOS_TOKEN = 'eos'
UNK_TOKEN = 'unk'
SOS_TOKEN_IDX = 0
EOS_TOKEN_IDX = 1

NEED_LOAD = False

ENC_DUMP_PATH = 'encoder.binary'
DEC_DUMP_PATH = 'decoder.binary'