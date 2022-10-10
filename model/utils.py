import numpy as np
import torch
import torch.nn as nn
import math

from config import SAMPLING_HYPARAMETER

def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs


def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word


def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


def sampling(logit, token_type):
    """
    讲解：https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277
    """

    t, p = SAMPLING_HYPARAMETER[token_type]['t'], SAMPLING_HYPARAMETER[token_type]['p']

    logit = logit.squeeze().detach().cpu().numpy()
    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word


class Embeddings(nn.Module):
    def __init__(self, n_token, D):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, D)
        self.D = D

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.D)


class PositionalEncoding(nn.Module):
    def __init__(self, D, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, D, 2).float() * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (?, seq_len, D)
        # pe: (1, max_len, D)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
