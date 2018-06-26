import argparse
import pickle
import time

import torch
import torchtext
from tensorboardX import SummaryWriter
from torch import optim, nn
from torch.utils.data.dataloader import DataLoader

import cfg
import opts
from dataset import LMDataset, Vocab
from modules.generator import Generator
from trainer import Trainer
from utils import time_since

# Instantiate parser
parser = argparse.ArgumentParser(description='main.py')
opts.model_opts(parser)
opts.training_opts(parser)
opts.model_io_opts(parser)
opts.data_opts(parser)

opt = parser.parse_args()
print('Arguments parser')
print(opt)

# Initialize all except model
# vocab = torchtext.vocab.GloVe(name='840B', dim='300', cache='/media/data/nlp/wv/glove')
vocab = pickle.load(open(opt.vocab_path, 'rb'))
corpus = pickle.load(open(opt.data_path, 'r'))
lmdataset = LMDataset(vocab=vocab, data=corpus)
qaloader = DataLoader(lmdataset, batch_size=opt.batch_size, shuffle=False)

if opt.tensorboard:
    writer = SummaryWriter(log_dir=opt.log_file_path)


# Initialize model
if opt.checkpoint:
    generator = torch.load(opt.checkpoint)
    # decoder = torch.load(cfg.DEC_DUMP_PATH)
    print('Successfully loaded from disk')
else:
    generator = Generator(
        opt.vocab_size if cfg.model.vocab_size > 0 else len(vocab.d),
        opt.embedding_size,
        opt.hidden_size,
    )
    print('Initialized new models')

generator.to(cfg.device)


# Initialize optimizers and criterion

generator_optimizer = optim.Adam(generator.parameters(), lr=opt.learning_rate)
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

    for epoch in range(1, opt.n_epochs + 1):
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
                    time_since(start, epoch / opt.n_epochs),
                    epoch,
                    epoch / opt.n_epochs * 100,
                    print_loss_avg
                )
                print(print_summary)

            if n_instances is not None:
                if idx > n_instances:
                    break
        if opt.tensorboard:
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
