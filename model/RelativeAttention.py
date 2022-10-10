import numpy as np
from math import sqrt
import torch
import torch.nn.functional as F
from torch.nn import Dropout, Module

from fast_transformers.attention_registry import AttentionRegistry, Optional, Float, EventDispatcherInstance
from fast_transformers.events import EventDispatcher, AttentionEvent


def sequence_mask(length, max_length=None):
    """Tensorflow의 sequence_mask를 구현"""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class RelativeAttention(Module):
    """Implement the scaled dot product attention with softmax.

    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, softmax_temp=None, attention_dropout=0.1,
                 event_dispatcher=""):
        super(RelativeAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

        self.max_seq = None
        self.E = None
        self.dh = None

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        """Implements the multihead softmax attention.

        Arguments
        ---------
            queries: (N, L, H, E) The tensor containing the queries
            keys: (N, S, H, E) The tensor containing the keys
            values: (N, S, H, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        """
        # Extract some shapes and compute the temperature
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        softmax_temp = self.softmax_temp or 1./sqrt(E)

        if self.max_seq is None:
            self.max_seq = L
            self.dh = E
            self.E = torch.randn([self.max_seq, int(self.dh)], requires_grad=False)

        # Scale the queries instead of applying the softmax temperature to the
        # dot products
        queries = queries * softmax_temp

        # Compute the unnormalized attention and apply the masks
        # (N, H, L, L)
        # QK = torch.einsum("nlhe,nshe->nhls", queries, keys)

        # TODO: 修改原来的FullAttention操作
        q = queries.permute(0, 2, 1, 3)
        k = keys.permute(0, 2, 1, 3)
        v = values.permute(0, 2, 1, 3)

        self.len_q = L
        self.len_k = S

        E = self._get_left_embedding(self.len_q, self.len_k).to(q.device)
        QE = torch.einsum('bhld,md->bhlm', [q, E])
        QE = self._qe_masking(QE)
        Srel = self._skewing(QE)

        Kt = k.permute(0, 1, 3, 2)
        QKt = torch.matmul(q, Kt)
        QK = QKt + Srel
        
        if not attn_mask.all_ones:
            QK = QK + attn_mask.additive_matrix
        if not key_lengths.all_ones:
            QK = QK + key_lengths.additive_matrix[:, None, None]

        # Compute the attention and the weighted average
        A = self.dropout(torch.softmax(QK, dim=-1))
        V = torch.einsum("nhls,nshd->nlhd", A, values)

        # Let the world know of the attention matrix
        self.event_dispatcher.dispatch(AttentionEvent(self, A))

        # print('N = {}, L = {}, H = {}, E = {}'.format(N, L, H, E))
        # print('S = {}, D = {}'.format(S, D))
        # print('QK: ', QK.shape)
        # print('A: ', A.shape)
        # print('V: ', V.shape)
        # print('V.contiguous(): ', V.contiguous().shape)

        # Make sure that what we return is contiguous
        return V.contiguous()

    def _get_left_embedding(self, len_q, len_k):
        starting_point = max(0, self.max_seq-len_q)
        e = self.E[starting_point:, :]
        return e

    def _skewing(self, tensor: torch.Tensor):
        padded = F.pad(tensor, [1, 0, 0, 0, 0, 0, 0, 0])
        reshaped = torch.reshape(padded, shape=[padded.size(
            0), padded.size(1), padded.size(-1), padded.size(-2)])
        Srel = reshaped[:, :, 1:, :]
        if self.len_k > self.len_q:
            Srel = F.pad(Srel, [0, 0, 0, 0, 0, 0, 0, self.len_k-self.len_q])
        elif self.len_k < self.len_q:
            Srel = Srel[:, :, :, :self.len_k]

        return Srel

    @staticmethod
    def _qe_masking(qe):
        mask = sequence_mask(
            torch.arange(qe.size()[-1] - 1, qe.size()
                         [-1] - qe.size()[-2] - 1, -1).to(qe.device),
            qe.size()[-1])
        mask = ~mask.to(mask.device)
        return mask.to(qe.dtype) * qe


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "relative", RelativeAttention,
    [
        ("softmax_temp", Optional(Float)),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
