import torch
from torch.autograd import Variable
from decoder import AttnDecoderRNN
from encoder import EncoderRNN


# ------------------------- ENCODER ----------------------
encoder_test = EncoderRNN(
    input_size=31, hidden_size=10, n_layers=1)
print(encoder_test)

encoder_hidden = encoder_test.init_hidden().cpu()
word_input = Variable(torch.LongTensor([1, 2, 3]).view(1, -1))  # (N, length) where N is mini-batch size
# if USE_CUDA:
#     encoder_test.cuda()
#     word_input = word_input.cuda()
encoder_outputs, encoder_hidden = encoder_test(word_input, encoder_hidden)

# ------------------------- DECODER ----------------------
decoder_test = AttnDecoderRNN(attn_model='general', embed_size=31, hidden_size=10, output_size=31, n_layers=1)
word_inputs = Variable(torch.LongTensor([1, 2, 3]).view(1, -1))
decoder_attns = torch.zeros(1, 3, 3)
decoder_hidden = encoder_hidden
decoder_context = Variable(torch.zeros(1, decoder_test.hidden_size))

# if USE_CUDA:
#     decoder_test.cuda()
#     word_inputs = word_inputs.cuda()
#     decoder_context = decoder_context.cuda()

for i in range(3):
    decoder_output, decoder_context, decoder_hidden, decoder_attn = decoder_test(word_inputs[i], decoder_context, decoder_hidden, encoder_outputs)
    print(decoder_output.size(), decoder_hidden.size(), decoder_attn.size())
    decoder_attns[0, i] = decoder_attn.squeeze(0).cpu().data
