import time
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from cfg import USE_CUDA
from dataset import QADataset
from decoder import AttnDecoderRNN
from encoder import EncoderRNN
from trainer import Trainer

import torchtext

from utils import time_since

vocab = torchtext.vocab.GloVe(name='840B', dim='300', cache='/media/data/nlp/wv/glove')

import pickle
final_data = pickle.load(open('/home/phobos_aijun/pytorch-experiments/DrQA/qa_final_data.pickle', 'rb'))

qadataset = QADataset(vocab=vocab, data = final_data)

qaloader = DataLoader(qadataset, batch_size=1, shuffle=False)

# TODO: organize config
attn_model = 'general'
hidden_size = 500
n_layers = 2
dropout_p = 0.1

# Initialize models
encoder = EncoderRNN(20000, hidden_size, n_layers)
decoder = AttnDecoderRNN(attn_model, hidden_size, 20000, n_layers, dropout_p=dropout_p)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

# Initialize optimizers and criterion
learning_rate = 0.0001
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()



# Configuring training
n_epochs = 1
plot_every = 200
print_every = 1000

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every

# Begin!
trainer = Trainer()
def main():
    for epoch in range(1, n_epochs + 1):
        for idx, batch in enumerate(qaloader):
            # Get training data for this cycle
            training_pair = batch
            input_variable = Variable(training_pair[0])  # 1 x len(training_pair[0])
            target_variable = Variable(training_pair[1])

            # Run the train function
            loss = trainer.train(
                input_variable,
                target_variable,
                encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                criterion
            )

            # Keep track of loss
            print_loss_total += loss
            plot_loss_total += loss

            if epoch == 0: continue

            if epoch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print_summary = '%s (%d %d%%) %.4f' % (
                time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
                print(print_summary)

            if epoch % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

if __name__ == '__main__':
    main()