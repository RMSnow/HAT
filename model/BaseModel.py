import token
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from fast_transformers.builders import TransformerEncoderBuilder, TransformerDecoderBuilder
from fast_transformers.builders import RecurrentEncoderBuilder, RecurrentDecoderBuilder
from fast_transformers.masking import TriangularCausalMask

from fast_transformers.recurrent.transformers import RecurrentTransformerEncoder, RecurrentTransformerEncoderLayer
from fast_transformers.attention import AttentionLayer
from RelativeAttention import RelativeAttention

from config import PREDICITED_TYPES, SAMPLING_HYPARAMETER
from utils import sampling, Embeddings, PositionalEncoding


class BaseModel(nn.Module):
    def __init__(self, args, dictionary):
        super(BaseModel, self).__init__()

        # Config
        self.args = args
        self.training = args.training
        self.D = args.d_model
        self.positon_dropout = args.position_dropout

        self.song_transformer_n_layer = args.song_transformer_n_layer
        self.song_transformer_n_head = args.song_transformer_n_head
        self.song_transformer_mlp = args.song_transformer_mlp
        self.song_transformer_dropout = args.song_transformer_dropout

        """
        type: 6 -> 64
        bar: 8 -> 64
        pos: 17 -> 128
        tempo: 94 -> 256
        structure: 15 -> 256
        chord: 322 -> 512
        track: 4 -> 64
        pitch: 83 -> 512
        duration: 81 -> 512 
        """
        self.event2word, self.word2event = dictionary

        n_class_dict = dict()
        for key in PREDICITED_TYPES:
            n_class_dict[key] = len(self.event2word[key])
        self.event_class_num_dict = n_class_dict

        self.emb_sizes = {'type': 64, 'bar': 64, 'pos': 128, 'tempo': 256, 'structure': 256,
                          'chord': 512, 'track': 64, 'pitch': 512, 'duration': 512}

        print('='*20)
        print('Predictied Type: \n')
        for k, v in self.event_class_num_dict.items():
            print('[{}] class_num = {}, embedding dim = {}'.format(
                k, v, self.emb_sizes[k]))
        print('='*20)

        # Embedding
        self.token_embeddings = []
        for k in PREDICITED_TYPES:
            self.token_embeddings.append(Embeddings(
                self.event_class_num_dict[k], self.emb_sizes[k]))
        self.token_embeddings = nn.ModuleList(self.token_embeddings)

        # MLP
        self.EncoderEmbLinear = nn.Linear(
            sum(list(self.emb_sizes.values())), self.D)
        # self.DecoderEmbLinear = nn.Linear(
        #     sum(list(self.emb_sizes.values())), self.D)
        self.TypeLinear = nn.Linear(self.D + self.emb_sizes['type'], self.D)

        # Position Embedding
        self.PosEmb = PositionalEncoding(
            self.D, dropout=self.positon_dropout)

        # Song Transformer
        # if args.song_transformer_is_recurrent:
        if not args.training:
            SongTransformerBuilder = RecurrentEncoderBuilder
        else:
            SongTransformerBuilder = TransformerEncoderBuilder

        if not args.training and args.transformer_attention_type == 'relative':
            self.SongTransformer = RecurrentTransformerEncoder([
                RecurrentTransformerEncoderLayer(
                    AttentionLayer(RelativeAttention(), self.D,
                                   self.song_transformer_n_head),
                    self.D,
                    d_ff=self.song_transformer_mlp,
                    activation='gelu',
                    dropout=self.song_transformer_dropout
                ) for _ in range(self.song_transformer_n_layer)
            ])

        self.SongTransformer = SongTransformerBuilder.from_kwargs(
            n_layers=self.song_transformer_n_layer,
            n_heads=self.song_transformer_n_head,
            query_dimensions=self.D//self.song_transformer_n_head,
            value_dimensions=self.D//self.song_transformer_n_head,
            feed_forward_dimensions=self.song_transformer_mlp,
            activation='gelu',
            attention_type=args.transformer_attention_type,
            dropout=self.song_transformer_dropout,
        ).get()

        # Project
        self.project_linears = []
        for k in PREDICITED_TYPES:
            self.project_linears.append(
                nn.Linear(self.D, self.event_class_num_dict[k]))
        self.project_linears = nn.ModuleList(self.project_linears)

        # Loss Function
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def forward_emb(self, x, EmbLinear):
        # x: (..., #event)

        # === Encoding: Embedding + MLP + Position Embedding === #
        x_event_embs = []
        for i, k in enumerate(PREDICITED_TYPES):
            x_event_embs.append(self.token_embeddings[i](x[..., i]))

        # (..., sum(embs))
        x_event_embs = torch.cat(x_event_embs, dim=-1)
        # print('x_event_embs: ', x_event_embs.shape)

        # (..., D)
        x_event = EmbLinear(x_event_embs)
        return x_event

    def forward_training(self, x, y_gt):
        """
        Params:
        # x: (bs, max_seq, #event)
        # y_gt (ground truth for x): (bs, max_seq, #event)

        Return:
        # y_events: a list whose size is #event. Every item is (bs, max_seq, event_class_num)
        """

        # 所有transformer的部分，加入mask
        bs, max_seq, events_num = x.size()
        song_seq_mask = TriangularCausalMask(max_seq, device=x.device)

        # === Encoding: Embedding + MLP + Position Embedding === #
        # (bs, max_seq, D)
        x_event = self.forward_emb(x, EmbLinear=self.EncoderEmbLinear)
        # print('x_event: ', x_event.shape)

        # === Song Transformer === #
        # Position Embedding
        x_song = self.PosEmb(x_event)
        # (bs, max_seq, D)
        x_song = self.SongTransformer(x_song, attn_mask=song_seq_mask)
        # print('x_song: ', x_song.shape)

        # === Prediction === #
        # (bs, max_seq, emb(type))
        y_gt_type_emb = self.token_embeddings[0](y_gt[:, :, 0])
        # (bs, max_seq, D)
        y_concat_gt_type = self.TypeLinear(
            torch.cat([x_song, y_gt_type_emb], dim=-1))

        y_events = []
        for i, k in enumerate(PREDICITED_TYPES):
            if i == 0:
                # (bs, max_seq, #type)
                y_event = self.project_linears[0](x_song)

            else:
                # (bs, max_seq, event_class_num)
                y_event = self.project_linears[i](y_concat_gt_type)

            y_events.append(y_event)

        return y_events

    def forward_generating_recurrently(self, x, song_transformer_state):
        """
        Params:
        # x: (1, #event)
        # song_transformer_state: a list

        Return:
        # y: (1, #event)
        # song_transformer_state
        """

        # === Encoding: Embedding + MLP + Position Embedding === #
        # (1, D)
        x_event = self.forward_emb(x, EmbLinear=self.EncoderEmbLinear)
        # print('x_event: ', x_event.shape)

        # === Song Transformer === #
        # x_song: (1, D), song_transformer_state: a list
        x_song, song_transformer_state = self.SongTransformer(
            x_event, state=song_transformer_state)

        # === Prediction === #
        # (1, #type)
        y_pred_type = self.project_linears[0](x_song)

        # Sampling
        y_pred_type_sampling = sampling(y_pred_type[0], token_type='type')
        # (1)
        y_pred_type = torch.as_tensor(
            y_pred_type_sampling, device=x.device).unsqueeze(-1)

        # (1, emb(type))
        y_pred_type_emb = self.token_embeddings[0](y_pred_type)

        # (1, D)
        # print('y_pred_type:', y_pred_type, y_pred_type.shape)
        # print('x_song:', x_song.shape, 'y_pred_type_emb:', y_pred_type_emb.shape)
        y_concat_pred_type = self.TypeLinear(
            torch.cat([x_song, y_pred_type_emb], dim=-1))

        y_preds = []
        for j, k in enumerate(PREDICITED_TYPES):
            if j == 0:
                # (1)
                y_preds.append(y_pred_type)
            else:
                # (1, event_class_num)
                y_pred_event = self.project_linears[j](y_concat_pred_type)

                # Sampling
                y_pred_event_sampling = sampling(y_pred_event[0], token_type=k)
                # (1)
                y_pred_event = torch.as_tensor(
                    y_pred_event_sampling, device=x.device).unsqueeze(-1)

                y_preds.append(y_pred_event)

            # print(k, y_preds[j].shape)

        # (1, #event)
        y_preds = torch.cat([y_.unsqueeze(-1) for y_ in y_preds], dim=-1)
        # print('y_preds: ', y_preds.shape)
        return y_preds, song_transformer_state

    def forward(self, x, y_gt=None, song_transformer_state=None):
        if self.training == True:
            return self.forward_training(x, y_gt)
        else:
            return self.forward_generating_recurrently(x, song_transformer_state)

    def compute_loss(self, pred_events, y_gt, loss_mask):
        # pred_events: a list whose size is #event. Every item is (bs, max_seq, event_class_num)
        # y_gt: (bs, max_seq, #event)
        # loss_mask: (bs, max_seq)

        event_losses = []
        for i, y_pred in enumerate(pred_events):
            # (bs, max_seq, event_class_num) -> (bs, event_class_num, max_seq)
            y_pred = y_pred.permute(0, -1, 1)
            # (bs, max_seq)
            loss = self.loss_func(y_pred, y_gt[..., i])

            loss = torch.sum(loss * loss_mask) / torch.sum(loss_mask)
            event_losses.append(loss)

        return event_losses
