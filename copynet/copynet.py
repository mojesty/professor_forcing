import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import time


class CopyEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(CopyEncoder, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)

        self.gru = nn.GRU(input_size=embed_size,
                          hidden_size=hidden_size, batch_first=True,
                          bidirectional=True)

    def forward(self, x):
        # input: [b x seq]
        embedded = self.embed(x)
        out, h = self.gru(embedded)  # out: [b x seq x hid*2] (biRNN)
        return out, h


class CopyDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(CopyDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.time = time.time()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(input_size=embed_size + hidden_size * 2,
                          hidden_size=hidden_size, batch_first=True)

        # weights
        self.Ws = nn.Linear(hidden_size * 2, hidden_size)  # only used at initial stage
        self.Wo = nn.Linear(hidden_size, vocab_size)  # generate mode
        self.Wc = nn.Linear(hidden_size * 2, hidden_size)  # copy mode
        self.nonlinear = nn.Tanh()

    def forward(self, input_idx, encoded, encoded_idx, prev_state, weighted, order):
        # input_idx(y_(t-1)): [b]			<- idx of next input to the decoder (Variable)
        # encoded: [b x seq x hidden*2]		<- hidden states created at encoder (Variable)
        # encoded_idx: [b x seq]			<- idx of inputs used at encoder (numpy)
        # prev_state(s_(t-1)): [1 x b x hidden]		<- hidden states to be used at decoder (Variable)
        # weighted: [b x 1 x hidden*2]		<- weighted attention of previous state, init with all zeros (Variable)

        # hyperparameters
        start = time.time()
        b = encoded.size(0)  # batch size
        seq_len = encoded.size(1)  # input sequence length
        vocab_size = self.vocab_size
        hidden_size = self.hidden_size

        # 0. set initial state s0 and initial attention (blank)
        if order == 0:
            prev_state = self.Ws(encoded[:, -1])
            weighted = torch.Tensor(b, 1, hidden_size * 2).zero_()
            weighted = self.to_cuda(weighted)
            weighted = Variable(weighted)

        prev_state = prev_state.unsqueeze(0)  # [1 x b x hidden]

        # 1. update states
        gru_input = torch.cat([self.embed(input_idx).unsqueeze(1), weighted], 2)  # [b x 1 x (h*2+emb)]
        _, state = self.gru(gru_input, prev_state)
        state = state.squeeze()  # [b x h]

        # 2. predict next word y_t
        # 2-1) get scores score_g for generation- mode
        score_g = self.Wo(state)  # [b x vocab_size]

        # 2-2) get scores score_c for copy mode, remove possibility of giving attention to padded values
        score_c = F.tanh(self.Wc(encoded.contiguous().view(-1, hidden_size * 2)))  # [b*seq x hidden_size]
        score_c = score_c.view(b, -1, hidden_size)  # [b x seq x hidden_size]
        score_c = torch.bmm(score_c, state.unsqueeze(2)).squeeze()  # [b x seq]
        encoded_mask = torch.Tensor(np.array(encoded_idx == 0, dtype=float) * (-1000))  # [b x seq]
        encoded_mask = self.to_cuda(encoded_mask)
        encoded_mask = Variable(encoded_mask)
        score_c = score_c + encoded_mask  # padded parts will get close to 0 when applying softmax
        score_c = F.tanh(score_c)  # purely optional....
        # 2-3) get softmax-ed probabilities
        score = torch.cat([score_g, score_c], 1)  # [b x (vocab+seq)]
        probs = F.softmax(score)
        prob_g = probs[:, :vocab_size]  # [b x vocab]
        prob_c = probs[:, vocab_size:]  # [b x seq]
        # remove scores which are obsolete

        # 2-4) add prob_c to prob_g
        prob_c_to_g = self.to_cuda(torch.Tensor(b, vocab_size).zero_())
        prob_c_to_g = Variable(prob_c_to_g)
        for b_idx in range(b):  # for each sequence in batch
            for s_idx in range(seq_len):
                prob_c_to_g[b_idx, encoded_idx[b_idx, s_idx]] = prob_c_to_g[b_idx, encoded_idx[b_idx, s_idx]] + prob_c[
                    b_idx, s_idx]

        # prob_c_to_g = Variable
        # en = torch.LongTensor(encoded_idx)
        # en.unsqueeze_(2)
        # one_hot = torch.FloatTensor(en.size(0),en.size(1),vocab_size).zero_()
        # one_hot.scatter_(2,en,1) # one hot tensor: [b x seq x vocab]
        # one_hot = self.to_cuda(one_hot)
        # prob_c_to_g = torch.bmm(prob_c.unsqueeze(1),Variable(one_hot)) # [b x 1 x vocab]
        # prob_c_to_g = prob_c_to_g.squeeze() # [b x vocab]
        out = prob_g + prob_c_to_g
        out = out.unsqueeze(1)  # [b x 1 x vocab]

        # 3. get weighted attention to use for predicting next word
        # 3-1) get tensor that shows whether each decoder input has previously appeared in the encoder
        idx_from_input = []
        for i, j in enumerate(encoded_idx):
            idx_from_input.append([int(k == input_idx[i].data[0]) for k in j])
        idx_from_input = torch.Tensor(np.array(idx_from_input, dtype=float))
        # idx_from_input : np.array of [b x seq]
        idx_from_input = self.to_cuda(idx_from_input)
        idx_from_input = Variable(idx_from_input)
        for i in range(b):
            if idx_from_input[i].sum().data[0] > 1:
                idx_from_input[i] = idx_from_input[i] / idx_from_input[i].sum().data[0]
        # 3-2) multiply with prob_c to get final weighted representation
        attn = prob_c * idx_from_input
        # for i in range(b):
        # 	tmp_sum = attn[i].sum()
        # 	if (tmp_sum.data[0]>1e-6):
        # 		attn[i] = attn[i] / tmp_sum.data[0]
        attn = attn.unsqueeze(1)  # [b x 1 x seq]
        weighted = torch.bmm(attn, encoded)  # weighted: [b x 1 x hidden*2]
        return out, state, weighted

    def to_cuda(self, tensor):
        # turns to cuda
        if torch.cuda.is_available():
            return tensor.cuda()
        else:
            return tensor

    def elapsed_time(self):
        elapsed = time.time()
        print("Time difference from prev. state: ", elapsed - self.time)
        self.time = elapsed
        return
