import pickle
import time

import torch
import torchtext
from tensorboardX import SummaryWriter
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

import cfg
from dataset import LMDataset, Vocab
from modules.generator import Generator
from trainer import Trainer
from utils import time_since

# vocab = torchtext.vocab.GloVe(name='840B', dim='300', cache='/media/data/nlp/wv/glove')
vocab = pickle.load(open('vocab.pt', 'rb'))
corpus = pickle.load(open('data.pt', 'rb'))
lmdataset = LMDataset(vocab=vocab, data=corpus)
qaloader = DataLoader(lmdataset, batch_size=cfg.batch_size, shuffle=False)

writer = SummaryWriter(log_dir=cfg.LOGDIR)


# Initialize models (or load them from disk)
if cfg.NEED_LOAD:
    generator = torch.load(cfg.ENC_DUMP_PATH)
    # decoder = torch.load(cfg.DEC_DUMP_PATH)
    print('Successfully loaded from disk')
else:
    generator = Generator(
        cfg.model.vocab_size if cfg.model.vocab_size > 0 else len(vocab.d),
        cfg.model.embedding_size,
        cfg.model.hidden_size,
    )
    print('Initialized new models')

generator.to(cfg.device)


# Initialize optimizers and criterion

generator_optimizer = optim.Adam(generator.parameters(), lr=cfg.learning_rate)
# decoder_optimizer = optim.Adam(model.decoder.parameters(), lr=learning_rate)


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

    for epoch in range(1, cfg.n_epochs + 1):
        for idx, batch in enumerate(qaloader):
            # print(idx)
            # Get training data for this cycle
            training_pair = batch
            input_variable = training_pair

            # Run the train function
            loss = trainer.train(
                input_variable,
                generator,
                generator_optimizer,
                None,
                None
            )
            if idx % print_every == 0:
                losses.append(loss)
            writer.add_scalar(
                cfg.NAME,
                loss,
                (epoch - 1) * (len(lmdataset) if n_instances is None else n_instances) + idx
            )

            # Keep track of loss
            print_loss_total += loss
            plot_loss_total += loss

            if epoch == 0: continue

            if idx % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print_summary = '%s (%d %d%%) %.4f' % (
                    time_since(start, epoch / cfg.n_epochs),
                    epoch,
                    epoch / cfg.n_epochs * 100,
                    print_loss_avg
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
            torch.save(generator, cfg.ENC_DUMP_PATH.format(epoch))

    writer.close()

if __name__ == '__main__':
    main()
