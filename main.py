import pickle
import time

import torch
import torchtext
from tensorboardX import SummaryWriter
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

import cfg
from cfg import USE_CUDA, n_epochs
from cfg import model
from dataset import QADataset
from model import Translator
from modules.decoder import AttnDecoderRNN
from modules.encoder import EncoderRNN
from trainer import Trainer
from utils import time_since

vocab = torchtext.vocab.GloVe(name='840B', dim='300', cache='/media/data/nlp/wv/glove')
final_data = pickle.load(open('/home/phobos_aijun/pytorch-experiments/DrQA/qa_final_data.pickle', 'rb'))
qadataset = QADataset(vocab=vocab, data=final_data, gpu=USE_CUDA)
qaloader = DataLoader(qadataset, batch_size=cfg.batch_size, shuffle=False)

writer = SummaryWriter(log_dir=cfg.LOGDIR)


# Initialize models (or load them from disk)
if cfg.NEED_LOAD:
    encoder = torch.load(cfg.ENC_DUMP_PATH)
    decoder = torch.load(cfg.DEC_DUMP_PATH)
    print('Successfully loaded from disk')
else:
    encoder = EncoderRNN(
        cfg.model.vocab_size,
        model.embedding_size,
        model.hidden_size,
        model.n_layers,
        dropout_p=model.dropout_p
    )
    decoder = AttnDecoderRNN(
        model.attn_model,
        model.embedding_size,
        model.hidden_size,
        model.vocab_size,
        model.n_layers,
        dropout_p=model.dropout_p
    )
    print('Initialized new models')

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

# TODO: too hard
# TODO: turn off dropout when evaluating
model = Translator(0, 0, 0, 0, 0, 'general', encoder=encoder, decoder=decoder)

# Initialize optimizers and criterion
learning_rate = 0.00005
encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(model.decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()


# Configuring training
plot_every = 200
print_every = 10

# Begin!
trainer = Trainer()


def main(n_instances=None):
    # Keep track of time elapsed and running averages
    start = time.time()
    losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    for epoch in range(1, n_epochs + 1):
        for idx, batch in enumerate(qaloader):
            # print(idx)
            # Get training data for this cycle
            training_pair = batch
            input_variable = Variable(training_pair[0])  # 1 x len(training_pair[0])
            target_variable = Variable(training_pair[1])

            # Run the train function
            loss = trainer.train(
                input_variable,
                target_variable,
                model,
                encoder_optimizer,
                decoder_optimizer,
                criterion
            )
            if idx % print_every == 0:
                losses.append(loss)
            writer.add_scalar(
                cfg.NAME,
                loss,
                (epoch - 1) * (len(qadataset) if n_instances is None else n_instances) + idx
            )

            # Keep track of loss
            print_loss_total += loss
            plot_loss_total += loss

            if epoch == 0: continue

            if idx % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print_summary = '%s (%d %d%%) %.4f' % (
                    time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg
                )
                print(print_summary)

            if n_instances is not None:
                if idx > n_instances:
                    break

        with open(cfg.LOSSDIR, 'w') as f:
            f.write(','.join(['{:5.2}' for i in losses]))
            f.close()

        if cfg.NEED_SAVE:
            if cfg.save == 'all':
                pass
            elif cfg.save == 'last':
                epoch = 'last'  # we overwrite the iterable variable but it's okay
            else:
                raise NotImplementedError
            torch.save(encoder, cfg.ENC_DUMP_PATH.format(epoch))
            torch.save(decoder, cfg.DEC_DUMP_PATH.format(epoch))

    writer.close()

if __name__ == '__main__':
    main(n_instances=10)
