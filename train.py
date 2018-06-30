import argparse
import time
import logging
import datetime
from logging.config import dictConfig

import torch
# import torchtext
from torch.utils.data.dataloader import DataLoader

import cfg
import opts
from dataset import LMDataset
from model import LMGan
from trainer import Trainer
from utils import time_since

# Instantiate parser and parse args
parser = argparse.ArgumentParser(description='train.py')
opts.model_opts(parser)
opts.training_opts(parser)
opts.model_io_opts(parser)
opts.data_opts(parser)
opt = parser.parse_args()

dictConfig(cfg.logging_cfg)
print('Arguments:')
print(opt)

# check cuda
if opt.cuda and not torch.cuda.is_available():
    raise RuntimeError('Cannot train on GPU because cuda is not available')

device = 'cuda' if opt.cuda else 'cpu'
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)


# Initialize all except model
# vocab = torchtext.vocab.GloVe(name='840B', dim='300', cache='/media/data/nlp/wv/glove')
# vocab = pickle.load(open(opt.vocab_path, 'rb'))
lmdataset = LMDataset(
    vocab_path=opt.vocab_path,
    corpus_path=opt.data_path,
    bptt=opt.bptt,
    device=device,
    min_counts=opt.min_counts
)
opt.vocab_size = len(lmdataset.vocab)
opt.device = device
lmloader = DataLoader(lmdataset, batch_size=opt.batch_size, shuffle=True)

# prefix is added to model name and to tensorboard scalar name
start_time = str(datetime.datetime.now()).replace(' ', '_').replace(':', '_')[:-10]
prefix = 'vocab_{}.emb_{}.hidden_{}.lr_{}.start_time_{}'.format(
    opt.vocab_size,
    opt.embedding_size,
    opt.hidden_size,
    opt.learning_rate,
    start_time
)

if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir=opt.log_file_dir)


# Initialize model
if opt.checkpoint:
    model = torch.load(opt.checkpoint)
    if model.opt.vocab_size != opt.vocab_size:
        raise RuntimeError("""
        Size mismatch:
        Checkpoint vocabulary size {:7}
        Model vocabulary size {:7}
        """)
    print('Successfully loaded from disk')
else:
    model = LMGan(opt)
    print('Initialized new models')
model.device = device
model.to(device)


# Configuring training
plot_every = opt.plot_every
print_every = opt.print_every


# Begin!
trainer = Trainer(opt, model)


def main(n_instances=None):
    # Keep track of time elapsed and running averages
    start = time.time()
    losses = []
    print_nll_loss_total = 0  # Reset every print_every
    print_g_loss_total = 0
    print_d_loss_total = 0
    plot_loss_total = 0  # Reset every plot_every

    for epoch in range(1, opt.n_epochs + 1):
        for idx, batch in enumerate(lmloader):
            nll_loss, gen_loss, disc_loss = trainer.train(opt, batch)

            if idx % print_every == 0:
                losses.append(nll_loss)
            if opt.tensorboard:
                step_no = (epoch - 1) * (len(lmdataset) if n_instances is None else n_instances) + idx
                if opt.adversarial:
                    tag_dict = {
                        'nll_loss': nll_loss,
                        'generator_loss': gen_loss,
                        'discriminator_loss': disc_loss,
                    }
                else:
                    tag_dict = {'nll_loss': nll_loss}
                writer.add_scalars(prefix, tag_dict, step_no)
                # writer.export_scalars_to_json()  # possibly not required
            # Keep track of loss
            print_nll_loss_total += nll_loss
            print_g_loss_total += gen_loss
            print_d_loss_total += disc_loss
            plot_loss_total += nll_loss

            if epoch == 0: continue

            if idx % print_every == 0:
                print_summary = '%s (%d %d%%) nll %.4f generator %.4f discriminator %.4f' % (
                    time_since(start, epoch / opt.n_epochs),
                    epoch,
                    epoch / opt.n_epochs * 100,
                    print_nll_loss_total / print_every,
                    print_g_loss_total / print_every,
                    print_d_loss_total / print_every,
                )
                logging.info(print_summary)
                print_nll_loss_total = 0
                print_g_loss_total = 0
                print_d_loss_total = 0

            if n_instances is not None:
                if idx > n_instances:
                    break

            # with open(cfg.LOSSDIR, 'w') as f:
            #     f.write(','.join(['{:5.2}' for i in losses]))
            #     f.close()

        if not opt.not_save:
            # add epoch and loss info
            fname = opt.save_path + '.' + prefix + '.epoch{:2}'.format(epoch) + '.loss{:4.1}.pt'.format(nll_loss)
            torch.save(model, fname)

    writer.close()

if __name__ == '__main__':
    main()
