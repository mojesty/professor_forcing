from numpy import random

import torch
import torchtext
import pickle

from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable

import cfg
from cfg import MAX_LENGTH, USE_CUDA
from dataset import QADataset
from model import Translator

vocab = torchtext.vocab.GloVe(name='840B', dim='300', cache='/media/data/nlp/wv/glove')
final_data = pickle.load(open('/home/phobos_aijun/pytorch-experiments/DrQA/qa_final_data.pickle', 'rb'))
qadataset = QADataset(vocab=vocab, data=final_data, gpu=USE_CUDA)
qaloader = DataLoader(qadataset, batch_size=1, shuffle=False)


def evaluate(model, dataset_idx=None, sentence=None, **kwargs):
    """
    Runs inference pass ON A SINGLE SENTENCE ONLY
    :param sentence:
    :param max_length:
    :return: list of decoded words
    """
    sample_method = kwargs.get('sample_method', 'multinomial')
    assert sample_method in ['argmax', 'multinomial']
    if dataset_idx is not None:
        input = Variable(qadataset[dataset_idx][0].unsqueeze(0))  # 1 x instance_length
    elif sentence is not None:
        raise NotImplementedError
        input = Variable(qadataset.indexes_from_sentence(sentence.split()[:MAX_LENGTH]))
    else:
        raise ValueError('Either dataset idx from 0 to {} or sentence should be specified'.format(len(qadataset)))

    _, word_indices, from_attns, decoder_attentions = model(input, mode='inference', **kwargs)

    decoded_words = [('_' if from_attn else '') + qadataset.itos(idx) for from_attn, idx in zip(from_attns, word_indices)]

    return decoded_words, decoder_attentions[:len(word_indices), :len(word_indices)]

def main(model, i):
    print('------------------------------------------------')
    print('Sentence  {}'.format(qadataset.data[i]['context']))
    print('Question  {}'.format(qadataset.data[i]['question']))
    for j in range(5):
        words, _ = evaluate(
            model,
            dataset_idx=i,
            fill_unks='attn',
            sample_method='multinomial'
        )
        print('Generated {}'.format(' '.join(words)))

if __name__ == '__main__':
    epoch = 7
    print('Successfully loaded from disk')
    encoder = torch.load(cfg.ENC_DUMP_PATH.format(epoch))
    decoder = torch.load(cfg.DEC_DUMP_PATH.format(epoch))
    model = Translator(1, 1, 1, .1, 'general')
    del model.encoder
    del model.decoder
    model.encoder = encoder
    model.decoder = decoder
    for idx in range(15):
        idx = random.randint(0, len(qadataset))
        main(model, idx)

