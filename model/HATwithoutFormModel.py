import token
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from fast_transformers.builders import TransformerEncoderBuilder, TransformerDecoderBuilder
from fast_transformers.builders import RecurrentEncoderBuilder, RecurrentDecoderBuilder
from fast_transformers.masking import TriangularCausalMask, LengthMask

from fast_transformers.attention import full_attention

from config import PREDICITED_TYPES, SAMPLING_HYPARAMETER
from utils import sampling, Embeddings, PositionalEncoding


class HATwithoutFormModel(nn.Module):
    def __init__(self, args, dictionary):
        super(HATwithoutFormModel, self).__init__()

        # Config
        self.args = args
        self.training = args.training
        self.D = args.d_model
        self.positon_dropout = args.position_dropout

        self.song_transformer_n_layer = args.song_transformer_n_layer // 2
        self.song_transformer_n_head = args.song_transformer_n_head
        self.song_transformer_mlp = args.song_transformer_mlp
        self.song_transformer_dropout = args.song_transformer_dropout

        self.chord_progression_transformer_n_layer = args.chord_progression_transformer_n_layer
        self.chord_progression_transformer_n_head = args.chord_progression_transformer_n_head
        self.chord_progression_transformer_mlp = args.chord_progression_transformer_mlp
        self.chord_progression_transformer_dropout = args.chord_progression_transformer_dropout

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
        if not args.training:
            self.TransformerBuilder = RecurrentEncoderBuilder
        else:
            self.TransformerBuilder = TransformerEncoderBuilder

        self.BottomSongTransformer = self.TransformerBuilder.from_kwargs(
            n_layers=self.song_transformer_n_layer,
            n_heads=self.song_transformer_n_head,
            query_dimensions=self.D//self.song_transformer_n_head,
            value_dimensions=self.D//self.song_transformer_n_head,
            feed_forward_dimensions=self.song_transformer_mlp,
            activation='gelu',
            attention_type=args.transformer_attention_type,
            dropout=self.song_transformer_dropout,
        ).get()

        assert self.args.HAT_without_form == True

        self.ChordProgressionTransformer = self.TransformerBuilder.from_kwargs(
            n_layers=self.chord_progression_transformer_n_layer,
            n_heads=self.chord_progression_transformer_n_head,
            query_dimensions=self.D//self.chord_progression_transformer_n_head,
            value_dimensions=self.D//self.chord_progression_transformer_n_head,
            feed_forward_dimensions=self.chord_progression_transformer_mlp,
            activation='gelu',
            attention_type=args.transformer_attention_type,
            dropout=self.chord_progression_transformer_dropout,
        ).get()

        self.UpSongTransformer = self.TransformerBuilder.from_kwargs(
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

    def forward_training(self, x, y_gt, texture_index, texture_index_chord_len, texture_index_note_len):
        """
        Params:
        # x: (bs, max_seq, #event)
        # y_gt (ground truth for x): (bs, max_seq, #event)
        # texture_index: (bs, max_song_chord, max_note + 1), texture[:, :, 0] -> Chord Token Index
        # texture_index_chord_len: (bs)
        # texture_index_note_len: (bs, max_song_chord)

        Return:
        # y_events: a list whose size is #event. Every item is (bs, max_seq, event_class_num)
        """

        # 所有transformer的部分，加入mask
        bs, max_seq, _ = x.size()
        max_song_chord, max_note = texture_index.size(
            1), texture_index.size(2) - 1
        song_triangular_mask = TriangularCausalMask(max_seq, device=x.device)
        cp_triangular_mask = TriangularCausalMask(
            max_song_chord, device=x.device)

        # === Encoding: Embedding + MLP + Position Embedding === #
        # (bs, max_seq, D)
        x_event = self.forward_emb(x, EmbLinear=self.EncoderEmbLinear)
        # print('x_event: ', x_event.shape)

        # === Bottom Song Transformer === #
        # Position Embedding
        x_song = self.PosEmb(x_event)
        # (bs, max_seq, D)
        x_song = self.BottomSongTransformer(
            x_song, attn_mask=song_triangular_mask)
        # print('x_song: ', x_song.shape, (torch.isnan(x_song)).any())

        # === Update Chord Tokens === #
        # *** Find ***
        D = x_song.size(-1)

        # (bs, max_song_chord, D)
        chord_encode = torch.zeros(bs, max_song_chord, D, device=x.device)

        for i in range(bs):
            chord_encode[i] = x_song[i][texture_index[i, :, 0]]

        # *** Chord Progression Transformer ***
        # (bs, max_song_chord, D)
        cp_encode = chord_encode

        cp_len_mask = LengthMask(
            texture_index_chord_len, max_len=max_song_chord, device=x.device)
        cp_encode = self.PosEmb(cp_encode)
        cp_encode = self.ChordProgressionTransformer(
            cp_encode, attn_mask=cp_triangular_mask, length_mask=cp_len_mask)

        # *** Update ***
        # 第一个Chord Token的信息量不变，其余的加上chord progression信息
        chord_encode[:, 1:, :] += cp_encode[:, :-1, :]

        # *** Save ***
        for i in range(bs):
            song_chord_len = texture_index_chord_len[i]
            x_song[i][texture_index[i, :song_chord_len, 0]
                      ] = chord_encode[i, :song_chord_len]

        # print('x_song: ', x_song.shape, (torch.isnan(x_song)).any())

        # === Up Song Transformer === #
        # (bs, max_seq, D)
        x_song = self.PosEmb(x_song)
        x_song = self.UpSongTransformer(
            x_song, attn_mask=song_triangular_mask)

        if (torch.isnan(x_song)).any():
            print('cp_encode: ', cp_encode.shape,
                  (torch.isnan(cp_encode)).any())
            exit()

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

    def forward_generating_recurrently(self, x, recurrent_transformer_states, recurrent_transformer_hidden_results, curr_chord_encode, curr_seq_index=0, curr_chord_index=0):
        """
        Params:
        # x: (1, #event)
        # recurrent_transformer_states: a list
        #   bottom_song_transformer_state
        #   chord_progression_transformer_state
        #   form_transformer_state
        #   up_song_transformer_state
        # recurrent_transformer_hidden_results: a list
        #   hidden_cp_encode: (1, D)
        #   hidden_form_encode: (1, D)
        # curr_chord_encode: (1, D)

        Return:
        # y: (1, #event)
        # recurrent_transformer_states
        # recurrent_transformer_hidden_results
        # curr_chord_encode
        """

        bottom_song_transformer_state, chord_progression_transformer_state, form_transformer_state, up_song_transformer_state = recurrent_transformer_states
        hidden_cp_encode, hidden_form_encode = recurrent_transformer_hidden_results

        # === Encoding: Embedding + MLP + Position Embedding === #
        # (1, D)
        x_event = self.forward_emb(x, EmbLinear=self.EncoderEmbLinear)
        # print('x_event: ', x_event.shape)

        # === Bottom Song Transformer === #
        # pe: (1, MAX_LEN, D)
        x_song = x_event + self.PosEmb.pe[:, curr_seq_index]
        # (1, D)
        x_song, bottom_song_transformer_state = self.BottomSongTransformer(
            x_song, state=bottom_song_transformer_state)

        # === Update Chord & Structure Tokens === #
        # Init
        zero_vec = torch.zeros_like(
            x_song, device=x_song.device, dtype=x_song.dtype)
        if hidden_cp_encode is None:
            hidden_cp_encode = zero_vec.clone()
        if curr_chord_encode is None:
            curr_chord_encode = zero_vec.clone()

        x_type = self.word2event['type'][x[0, 0].item()]
        # print('x_type:', x_type)

        if x_type == 'Chord':
            # 更改curr_chord_encode
            curr_chord_encode = x_song

            # 融合Context Information
            x_song += hidden_cp_encode

            hidden_cp_encode, chord_progression_transformer_state = self.ChordProgressionTransformer(
                curr_chord_encode + self.PosEmb.pe[:, curr_chord_index], state=chord_progression_transformer_state)
            curr_chord_index += 1

        # === Up Song Transformer === #
        # (1, D)
        x_song += self.PosEmb.pe[:, curr_seq_index]
        x_song, up_song_transformer_state = self.UpSongTransformer(
            x_song, state=up_song_transformer_state)
        curr_seq_index += 1

        if (torch.isnan(x_song)).any():
            print('ERROR!')
            exit()

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

        # (1, #event)
        y_preds = torch.cat([y_.unsqueeze(-1) for y_ in y_preds], dim=-1)

        # print('y_preds:', y_preds)
        # exit()

        recurrent_transformer_states = [bottom_song_transformer_state,
                                        chord_progression_transformer_state, form_transformer_state, up_song_transformer_state]
        recurrent_transformer_hidden_results = [
            hidden_cp_encode, hidden_form_encode]

        return y_preds, recurrent_transformer_states, recurrent_transformer_hidden_results, curr_chord_encode, curr_seq_index, curr_chord_index

    def forward(self, x, y_gt=None, texture_index=None, texture_index_chord_len=None, texture_index_note_len=None, recurrent_transformer_states=None, recurrent_transformer_hidden_results=None, curr_chord_encode=None, curr_seq_index=None, curr_chord_index=None):
        if self.args.training:
            return self.forward_training(x, y_gt, texture_index, texture_index_chord_len, texture_index_note_len)
        else:
            return self.forward_generating_recurrently(x, recurrent_transformer_states, recurrent_transformer_hidden_results, curr_chord_encode, curr_seq_index, curr_chord_index)

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
