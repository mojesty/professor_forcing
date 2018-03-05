from numpy import random

import torch
import torchtext
import pickle

from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable

import cfg
from cfg import MAX_LENGTH, USE_CUDA, sos_idx, eos_idx, eos, unk_idx
from dataset import QADataset

vocab = torchtext.vocab.GloVe(name='840B', dim='300', cache='/media/data/nlp/wv/glove')
final_data = pickle.load(open('/home/phobos_aijun/pytorch-experiments/DrQA/qa_final_data.pickle', 'rb'))
qadataset = QADataset(vocab=vocab, data=final_data, gpu=USE_CUDA)
qaloader = DataLoader(qadataset, batch_size=1, shuffle=False)

def evaluate(
        encoder,
        decoder,
        dataset_idx=None,
        sentence=None,
        max_length=MAX_LENGTH,
        fill_unks=None,
        sample_mathod='argmax'
    ):
    """
    Runs inference pass ON A SINGLE SENTENCE ONLY
    :param sentence:
    :param max_length:
    :return: list of decoded words
    """
    # TODO: batch evaluation!
    assert sample_mathod in ['argmax', 'multinomial']
    if dataset_idx is not None:
        input_variable = Variable(qadataset[dataset_idx][0].unsqueeze(0))  # 1 x instance_length
    elif sentence is not None:
        raise NotImplementedError
        input_variable = Variable(qadataset.indexes_from_sentence(sentence.split()[:MAX_LENGTH]))
    else:
        raise ValueError('Either dataset idx from 0 to {} or sentence should be specified'.format(len(qadataset)))

    # Run through encoder
    encoder_hidden = encoder.init_hidden(batch_size=1)
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[sos_idx]]))  # SOS
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden

    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()


    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
            decoder_input,
            decoder_context,
            decoder_hidden,
            encoder_outputs
        )
        decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()

    return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]

def main(encoder, decoder, i):
    print('------------------------------------------------')
    print('Sentence  {}'.format(qadataset.data[i]['context']))
    print('Question  {}'.format(qadataset.data[i]['question']))
    for j in range(5):
        words, _ = evaluate(
            encoder=encoder,
            decoder=decoder,
            dataset_idx=i,
            max_length=MAX_LENGTH,
            fill_unks='attn',
            sample_mathod='multinomial'
        )
        print('Generated {}'.format(' '.join(words)))

if __name__ == '__main__':
    epoch = 7
    print('Successfully loaded from disk')
    encoder = torch.load(cfg.ENC_DUMP_PATH.format(epoch))
    decoder = torch.load(cfg.DEC_DUMP_PATH.format(epoch))
    for idx in range(15):
        idx = random.randint(0, len(qadataset))
        main(encoder, decoder, idx)
