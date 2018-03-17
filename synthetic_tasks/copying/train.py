import time

from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader


from synthetic_tasks.copying.dataset import CopyDataset
from synthetic_tasks.copying.model import CopyNet
from synthetic_tasks.copying import cfg
from utils import time_since

copydataset = CopyDataset(cfg.vocab_size, cfg.dataset.num_chars, cfg.dataset.num_pad)
copyloader = DataLoader(copydataset, batch_size=cfg.CopyModel.batch_size, shuffle=False)


model = CopyNet(
    cfg.vocab_size,
    cfg.CopyModel.hidden_size,
    cfg.vocab_size
)
if cfg.gpu:
    model.cuda()
print('Initialized new model')


encoder_optimizer = optim.RMSprop(
    [p for p in model.parameters() if p.requires_grad],
    lr=cfg.learning_rate
)
criterion = nn.NLLLoss()


# Configuring training
plot_every = 200
print_every = 10


def main():
    # Keep track of time elapsed and running averages
    start = time.time()
    losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    for epoch, batch in enumerate(copyloader):
        training_pair = batch
        input_variable = Variable(training_pair[0])  # [batch_size x len(training_pair[0])]
        target_variable = Variable(training_pair[1]).view(-1)
        batch_size = input_variable.size(0)
        model.zero_grad()
        model.hidden = model.init_hidden(batch_size)

        # Run the train function
        model_scores = model(input_variable)
        loss = criterion(model_scores, target_variable)
        loss.backward()
        encoder_optimizer.step()

        loss_value = loss.cpu().data.squeeze()[0]
        if epoch % print_every == 0:
            losses.append(loss_value)

        # Keep track of loss
        print_loss_total += loss_value
        plot_loss_total += loss_value

        if epoch == 0: continue

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (
                time_since(start, epoch / cfg.n_epochs), epoch, epoch / cfg.n_epochs * 100, print_loss_avg
            )
            print(print_summary)


if __name__ == '__main__':
    main()
