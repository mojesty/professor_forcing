import torch
import torchtext
import pickle

from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable

import cfg
from cfg import MAX_LENGTH, USE_CUDA, SOS_TOKEN_IDX, EOS_TOKEN_IDX, EOS_TOKEN, UNK_TOKEN_IDX
from dataset import QADataset

vocab = torchtext.vocab.GloVe(name='840B', dim='300', cache='/media/data/nlp/wv/glove')
final_data = pickle.load(open('/home/phobos_aijun/pytorch-experiments/DrQA/qa_final_data.pickle', 'rb'))
qadataset = QADataset(vocab=vocab, data=final_data, gpu=USE_CUDA)
qaloader = DataLoader(qadataset, batch_size=1, shuffle=False)

def evaluate(encoder, decoder, dataset_idx=None, sentence=None, max_length=MAX_LENGTH, fill_unks=None):
    """
    Runs inference pass
    :param sentence:
    :param max_length:
    :return:
    """
    if dataset_idx is not None:
        input_variable = Variable(qadataset[dataset_idx][0].unsqueeze(0))  # 1 x instance_length
    elif sentence is not None:
        raise NotImplementedError
        input_variable = Variable(qadataset.indexes_from_sentence(sentence.split()[:MAX_LENGTH]))
    else:
        raise ValueError('Either dataset idx from 0 to {} or sentence should be specified'.format(len(qadataset)))

    # Run through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[SOS_TOKEN_IDX]]))  # SOS
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden

    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()


    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length + 2)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                                     decoder_context,
                                                                                     decoder_hidden,
                                                                                     encoder_outputs)
        decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_TOKEN_IDX:
            decoded_words.append(EOS_TOKEN)
            break
        elif ni == UNK_TOKEN_IDX and fill_unks is not None:
            top_word, top_word_idx = decoder_attention.data.topk(1)
            decoded_words.append(qadataset.itos(input_variable.squeeze(0)[top_word_idx.view(-1)].cpu().data[0]))
        else:
            decoded_words.append(qadataset.itos(ni))

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()

    return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]

def main(i):
    encoder = torch.load(cfg.ENC_DUMP_PATH)
    decoder = torch.load(cfg.DEC_DUMP_PATH)
    print('Successfully loaded from disk')
    print('Sentence  {}'.format(qadataset.data[i]['context']))
    print('Question  {}'.format(qadataset.data[i]['question']))
    words, _ = evaluate(encoder=encoder, decoder=decoder, dataset_idx=i, max_length=MAX_LENGTH, fill_unks='attn')
    print('Generated {}'.format(words))

if __name__ == '__main__':
    for idx in range(3, 15):
        main(idx)
