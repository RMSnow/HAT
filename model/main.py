import pickle
import numpy as np
import os
import time
import random
import sys
import json

import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note
import pretty_midi

from config import SAMPLING_HYPARAMETER, parser, PREDICITED_TYPES, LOSSES_WEIGHTS
from BaseModel import BaseModel
from FormModel import FormModel
from HATwithoutTextureModel import HATwithoutTextureModel
from HATwithoutFormModel import HATwithoutFormModel
# from TextureModel import TextureModel
# from FormTextureModel import FormTextureModel
from SongDataset import SongDataset


BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4

CONTINUE = '<CONTI>'
UNKNOWN = '<UNK>'


def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def save_model(model, epoch, optimizer, save_dir, save_name):
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    },
        os.path.join(save_dir, '{}.pt'.format(save_name))
    )


def train():
    # Loading Data
    print('-'*10, 'Loading data...', '-'*10)
    start = time.time()

    path_dictionary = os.path.join(args.data_dir, 'dictionary.pkl')
    path_training_data = os.path.join(args.data_dir, 'train_data.npz')

    dictionary = pickle.load(open(path_dictionary, 'rb'))
    train_data = np.load(path_training_data, allow_pickle=True)

    if args.debug:
        train_song_dataset = SongDataset(
            train_data, sz=20 * args.batch_size, model_name=args.model)
    else:
        train_song_dataset = SongDataset(train_data, model_name=args.model)

    train_data_loader = DataLoader(
        train_song_dataset, batch_size=args.batch_size, shuffle=True)

    print('Done. It took {:.6f}s.'.format(time.time() - start))
    print()

    # Loading Model
    print('-'*10, 'Loading model...', '-'*10)
    start = time.time()

    device = args.device
    net = eval('{}(args, dictionary)'.format(args.model))

    # TODO: 多卡
    if args.parallel and torch.cuda.device_count() > 1:
        print('DataParallel!')
        net = nn.DataParallel(net)

    net.to(device)
    net.train()
    n_parameters = network_paras(net)
    args.n_parameters = n_parameters
    print('n_parameters: {:,}'.format(n_parameters))

    print('Done. It took {:.6f}s.'.format(time.time() - start))
    print()

    print(net)
    print()

    # optimizers
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    # Loss Weights
    print('losses_weights: ', LOSSES_WEIGHTS)
    losses_weights = list(LOSSES_WEIGHTS.values())
    print()

    # Save Configuration
    args_file = os.path.join(args.save, 'args.txt')
    with open(args_file, 'w') as f:
        f.write('[Arguments]\n')
        print_s = ''
        for arg in vars(args):
            s = '{}\t{}\n'.format(arg, getattr(args, arg))
            f.write(s)
            print_s += s

        f.write('\n\n[Model]\n')
        f.write('{}'.format(net))

        f.write('\n\n[Dataset]\n')
        f.write('train_x: {}\n'.format(train_song_dataset.x.shape))
        f.write('train_y: {}\n'.format(train_song_dataset.y.shape))
        f.write('train_mask: {}\n'.format(train_song_dataset.mask.shape))

        print('\n---------------------------------------------------\n')
        print('[Arguments] \n')
        print(print_s)
        print('\n---------------------------------------------------\n')

    # Training
    print('-'*10, 'Training...', '-'*10)

    # run
    max_grad_norm = 3

    start_time = time.time()
    for epoch in range(args.epochs):
        acc_loss = 0
        acc_losses = np.zeros(9)

        for bidx, batch_items in enumerate(train_data_loader):
            batch_x, batch_y, batch_mask = batch_items[:3]
            batch_x = batch_x.long().to(device)
            batch_y = batch_y.long().to(device)
            batch_mask = batch_mask.float().to(device)

            if args.model in ['FormModel', 'HATwithoutTextureModel']:
                batch_form_index, batch_form_index_section_len, batch_form_index_chord_len = batch_items[
                    3:]
                batch_form_index = batch_form_index.long().to(device)
                batch_form_index_section_len = batch_form_index_section_len.long().to(device)
                batch_form_index_chord_len = batch_form_index_chord_len.long().to(device)

                y_events = net(batch_x, y_gt=batch_y, form_index=batch_form_index,
                               form_index_section_len=batch_form_index_section_len, form_index_chord_len=batch_form_index_chord_len)

            elif args.model in ['TextureModel', 'HATwithoutFormModel']:
                batch_texture_index, batch_texture_index_chord_len, batch_texture_index_note_len = batch_items[
                    3:]
                batch_texture_index = batch_texture_index.long().to(device)
                batch_texture_index_chord_len = batch_texture_index_chord_len.long().to(device)
                batch_texture_index_note_len = batch_texture_index_note_len.long().to(device)

                y_events = net(batch_x, y_gt=batch_y, texture_index=batch_texture_index,
                               texture_index_chord_len=batch_texture_index_chord_len, texture_index_note_len=batch_texture_index_note_len)

            elif args.model == 'FormTextureModel':
                batch_form_index, batch_form_index_section_len, batch_form_index_chord_len, batch_texture_index, batch_texture_index_chord_len, batch_texture_index_note_len = batch_items[
                    3:]

                batch_form_index = batch_form_index.long().to(device)
                batch_form_index_section_len = batch_form_index_section_len.long().to(device)
                batch_form_index_chord_len = batch_form_index_chord_len.long().to(device)
                batch_texture_index = batch_texture_index.long().to(device)
                batch_texture_index_chord_len = batch_texture_index_chord_len.long().to(device)
                batch_texture_index_note_len = batch_texture_index_note_len.long().to(device)

                y_events = net(batch_x, y_gt=batch_y, form_index=batch_form_index,
                               form_index_section_len=batch_form_index_section_len, form_index_chord_len=batch_form_index_chord_len, texture_index=batch_texture_index,
                               texture_index_chord_len=batch_texture_index_chord_len, texture_index_note_len=batch_texture_index_note_len)

            elif args.model == 'BaseModel':
                y_events = net(batch_x, y_gt=batch_y)

            if args.parallel:
                losses = net.module.compute_loss(y_events, batch_y, batch_mask)
            else:
                losses = net.compute_loss(y_events, batch_y, batch_mask)

            loss = 0
            for i, weight in enumerate(losses_weights):
                loss += weight * losses[i]
            loss /= len(losses_weights)

            # Update
            net.zero_grad()
            loss.backward()

            if max_grad_norm is not None:
                clip_grad_norm_(net.parameters(), max_grad_norm)

            optimizer.step()

            # print
            sys.stdout.write('{}/{} | Loss: {:.6f} | type: {:.6f}, bar: {:.6f}, pos: {:.6f}, tempo: {:.6f}, structure {:.6f}, chord: {:.6f}, track: {:.6f}, pitch {:.6f}, duration {:.6f} \r'.format(bidx, len(train_data_loader), loss,
                                                                                                                                                                                                     losses[0] * losses_weights[0], losses[1] * losses_weights[1], losses[2] * losses_weights[2], losses[3] * losses_weights[3], losses[4] * losses_weights[4], losses[5] * losses_weights[5], losses[6] * losses_weights[6], losses[7] * losses_weights[7], losses[8] * losses_weights[8]))
            sys.stdout.flush()

            # acc
            acc_losses += np.array([l.item() for l in losses])
            acc_loss += loss.item()

        # epoch loss
        epoch_loss = acc_loss / len(train_data_loader)
        acc_losses = acc_losses / len(train_data_loader)
        print('\n', '-'*20, '\n')
        print('epoch: {}/{} | Loss: {} | time: {}'.format(
            epoch, args.epochs, epoch_loss, time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(time.time()))))
        each_loss_str = 'type: {:.6f}, bar: {:.6f}, pos: {:.6f}, tempo: {:.6f}, structure {:.6f}, chord: {:.6f}, track: {:.6f}, pitch {:.6f}, duration {:.6f} \r'.format(
            acc_losses[0] * losses_weights[0], acc_losses[1] * losses_weights[1], acc_losses[2] * losses_weights[2], acc_losses[3] * losses_weights[3], acc_losses[4] * losses_weights[4], acc_losses[5] * losses_weights[5], acc_losses[6] * losses_weights[6], acc_losses[7] * losses_weights[7], acc_losses[8] * losses_weights[8])
        print('    >', each_loss_str)
        print('\n', '-'*20, '\n')

        # save model, with policy
        loss = epoch_loss
        if 0.2 < loss <= 0.5:
            fn = int(loss * 10) * 10
            save_model(net, epoch, optimizer, args.save, 'loss_' + str(fn))
        elif 0.02 < loss <= 0.20:
            fn = int(loss * 100)
            save_model(net, epoch, optimizer, args.save, 'loss_' + str(fn))
        elif loss <= 0.02:
            fn = int(loss * 100)
            save_model(net, epoch, optimizer, args.save, 'loss_' + str(fn))
            exit()
        else:
            save_model(net, epoch, optimizer, args.save, 'loss_high')


def generate(num=10, prompt_file=''):
    model_name = args.model_file.split('/')[-1].split('.')[0]

    if not args.sampling_strategy_is_nucleus:
        model_name += '_weighted_sampling'
    if prompt_file != '':
        prompt_name = prompt_file.split('/')[-1].split('.')[0]
        model_name += '_' + prompt_name

    save_dir = os.path.join(args.save, model_name)
    save_midi_dir = os.path.join(save_dir, '{}_mid'.format(model_name))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        os.system('rm -r {}'.format(save_dir))
        os.mkdir(save_dir)
    if not os.path.exists(save_midi_dir):
        os.mkdir(save_midi_dir)

    # Loading Data
    path_dictionary = os.path.join(args.data_dir, 'dictionary.pkl')
    path_test_data = os.path.join(args.data_dir, 'test_data.npz')
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary
    test_data = np.load(path_test_data)['x']

    # Loading Model
    print('-'*10, 'Loading model...', '-'*10)
    start = time.time()

    n_class_dict = dict()
    for key in PREDICITED_TYPES:
        n_class_dict[key] = len(event2word[key])

    device = args.device
    net = eval('{}(args, dictionary)'.format(args.model))
    resume_dict = torch.load(args.model_file)['state_dict']

    # 对于多卡训练的模型，要修改resume_dict
    if args.parallel:
        new_resume_dict = dict()
        for k, v in resume_dict.items():
            if 'module.' in k:
                new_resume_dict[k.replace('module.', '')] = v
        resume_dict = new_resume_dict

    net.load_state_dict(resume_dict)

    net.to(device)
    net.eval()

    print('Done. It took {:.6f}s.'.format(time.time() - start))
    print()

    # Save Configuration
    args_file = os.path.join(save_dir, 'args.txt')
    with open(args_file, 'w') as f:
        f.write('[Arguments]\n')
        print_s = ''
        for arg in vars(args):
            s = '{}\t{}\n'.format(arg, getattr(args, arg))
            f.write(s)
            print_s += s

        f.write('\n\n[Model]\n')
        f.write('{}'.format(net))

        print('\n---------------------------------------------------\n')
        print('[Arguments] \n')
        print(print_s)
        print('\n---------------------------------------------------\n')

    # Generate
    print('-'*10, 'Generating...', '-'*10)
    for i in range(num):
        song_name = '{}_{}'.format(model_name, i)
        print('-'*10, 'Song: {}'.format(song_name), '-'*10)

        with torch.no_grad():
            song_transformer_state = None
            recurrent_transformer_states = [None, None, None, None]
            recurrent_transformer_hidden_results = [None, None]
            curr_chord_encode = None
            curr_section_encode = None
            curr_context_section_encode = None
            curr_context_chord_encode = None
            curr_seq_index, curr_chord_index, curr_section_index = [0, 0, 0]

            log_f = open(os.path.join(
                save_dir, '{}.log'.format(song_name)), 'w')
            bar_num = 0

            if prompt_file == '':
                # (1, #event)
                sos = torch.as_tensor(
                    test_data[0, 0], dtype=torch.long, device=device).unsqueeze(0)
                print('\n<SOS>:\n{}\n'.format(sos))

                # Results
                res = [sos]
                # Input
                x = sos

            else:
                # (#propmt, 1, #event)
                prompt = torch.as_tensor(
                    np.load(prompt_file), dtype=torch.long, device=device).unsqueeze(1)
                sos = prompt[0]
                print('\nPrompt: ', prompt.shape)
                print('\n<SOS>:\n{}\n'.format(sos))

                prompt_index = 0
                prompt_sz = len(prompt)
                res = [sos]

            while True:
                # x: (1, #event), y: (1, #event)

                if prompt_file != '' and prompt_index < prompt_sz:
                    x = prompt[prompt_index]
                    prompt_index += 1

                if args.model == 'FormTextureModel':
                    # x_type = word2event['type'][x[0, 0].item()]
                    # print('x_type: ', x_type)

                    # hidden_cp_encode, hidden_form_encode = recurrent_transformer_hidden_results
                    # if curr_section_encode is not None:
                    #     print('curr_section_encode:', curr_section_encode[0, :5])
                    # if curr_context_section_encode is not None:
                    #     print('curr_context_section_encode:', curr_context_section_encode[0, :5])
                    # if curr_context_chord_encode is not None:
                    #     print('curr_context_chord_encode: ', curr_context_chord_encode[0, :5])
                    # if hidden_cp_encode is not None:
                    #     print('hidden_cp_encode: ', hidden_cp_encode[0, :5])
                    # if hidden_form_encode is not None:
                    #     print('hidden_form_encode: ', hidden_form_encode[0, :5])
                    # print()

                    y, recurrent_transformer_states, recurrent_transformer_hidden_results, curr_section_encode, curr_context_section_encode, curr_context_chord_encode, curr_seq_index, curr_chord_index, curr_section_index = net(
                        x, recurrent_transformer_states=recurrent_transformer_states, recurrent_transformer_hidden_results=recurrent_transformer_hidden_results, curr_section_encode=curr_section_encode,
                        curr_context_section_encode=curr_context_section_encode,
                        curr_context_chord_encode=curr_context_chord_encode,
                        curr_seq_index=curr_seq_index,
                        curr_chord_index=curr_chord_index,
                        curr_section_index=curr_section_index)

                    # y_type = word2event['type'][y[0, 0].item()]
                    # print('y_type: ', y_type)
                    # print('curr_section_encode:', curr_section_encode[0, :5])
                    # print('curr_context_section_encode:', curr_context_section_encode[0, :5])
                    # print('curr_context_chord_encode: ', curr_context_chord_encode[0, :5])
                    # hidden_cp_encode, hidden_form_encode = recurrent_transformer_hidden_results
                    # print('hidden_cp_encode: ', hidden_cp_encode[0, :5])
                    # print('hidden_form_encode: ', hidden_form_encode[0, :5])
                    # print('-'*20)

                    # if len(res) == 10:
                    #     break

                if args.model in ['FormModel', 'HATwithoutTextureModel']:
                    y, recurrent_transformer_states, recurrent_transformer_hidden_results, curr_section_encode, curr_context_section_encode, curr_seq_index, curr_chord_index, curr_section_index = net(
                        x, recurrent_transformer_states=recurrent_transformer_states, recurrent_transformer_hidden_results=recurrent_transformer_hidden_results, curr_section_encode=curr_section_encode,
                        curr_context_section_encode=curr_context_section_encode,
                        curr_seq_index=curr_seq_index,
                        curr_chord_index=curr_chord_index,
                        curr_section_index=curr_section_index)

                if args.model == 'HATwithoutFormModel':
                    y, recurrent_transformer_states, recurrent_transformer_hidden_results, curr_chord_encode, curr_seq_index, curr_chord_index = net(
                        x, recurrent_transformer_states=recurrent_transformer_states, recurrent_transformer_hidden_results=recurrent_transformer_hidden_results, curr_chord_encode=curr_chord_encode,
                        curr_seq_index=curr_seq_index,
                        curr_chord_index=curr_chord_index)

                if args.model == 'BaseModel':
                    y, song_transformer_state = net(
                        x, song_transformer_state=song_transformer_state)

                if prompt_file != '' and prompt_index < prompt_sz:
                    y = prompt[prompt_index]

                x = y
                res.append(y)

                # (#note, #event)
                y = y.flatten(end_dim=-2)

                # Logging
                s = ''
                for note in y:
                    bar = word2event['bar'][note[1].item()]
                    if type(bar) == int:
                        s += '-' * 20 + 'Bar: {}'.format(bar_num) + '\n'
                        bar_num += bar

                    for k, w in enumerate(note):
                        s += '{:10s}'.format(
                            str(word2event[PREDICITED_TYPES[k]][w.item()])) + ' | '

                log_f.write(s + '\n')
                log_f.flush()

                if args.verbose:
                    print(s)

                # 判断结尾
                # if y[0, 1] == event2word['bar']['<EOS>'] or bar_num >= args.max_generated_bar or len(res) >= args.max_seq_len:
                #     break

                if y[0, 1] == event2word['bar']['<EOS>'] or len(res) >= args.max_seq_len:
                    break

        # (#total_note, #event)
        res = torch.cat(res, dim=0).detach().cpu().numpy()
        np.save(os.path.join(save_dir, '{}_{}.npy'.format(
            song_name, res.shape)), res)

        res_tokens = []
        for note in res:
            token = {k: word2event[k][note[i]]
                     for i, k in enumerate(PREDICITED_TYPES)}
            res_tokens.append(token)
        with open(os.path.join(save_dir, '{}_{}.json'.format(song_name, len(res_tokens))), 'w') as f:
            json.dump(res_tokens, f, indent=4, ensure_ascii=False)

        write2midi(res, word2event, os.path.join(
            save_midi_dir, '{}.mid'.format(song_name)))


def write2midi(arr, word2event, path_outfile, write_structure=True, write_chord=False):
    # arr: (#note, #event)

    # arr to words
    words = []
    for note in arr:
        token = dict()
        for i, k in enumerate(PREDICITED_TYPES):
            token[k] = word2event[k][note[i]]
        words.append(token)

    midi_obj = miditoolkit.midi.parser.MidiFile()

    all_notes = [[], [], []]
    track2index = {'MELODY': 0, 'BRIDGE': 1, 'PIANO': 2}
    velocity_dict = {'MELODY': 80, 'BRIDGE': 80, 'PIANO': 60}

    bar_cnt = -1

    for i in range(len(words)):
        token = words[i]

        if token['type'] in ['EOS_CHORD', 'EOS_SIMUNOTE', 'PAD']:
            continue

        if token['bar'] == '<SOS>':
            continue
        elif token['bar'] == '<EOS>':
            continue
            # break

        # Bar: 0, num, <CONTI>
        if token['bar'] != CONTINUE:
            bar_cnt += token['bar']
        # Beat_?
        beat_pos = int(token['pos'].split('_')[-1])
        curr_pos = bar_cnt * BAR_RESOL + beat_pos * TICK_RESOL

        if token['type'] == 'Boundary':
            pass

        elif token['type'] == 'Tempo':
            if token['tempo'] not in [0, CONTINUE, UNKNOWN]:
                tempo = int(token['tempo'])
                midi_obj.tempo_changes.append(
                    TempoChange(tempo=tempo, time=curr_pos))

        elif token['type'] == 'Structure':
            if write_structure and token['structure'] not in [0, CONTINUE, UNKNOWN]:
                midi_obj.markers.append(
                    Marker(text=token['structure'], time=curr_pos))

        elif token['type'] == 'Chord':
            if write_chord and token['chord'] not in [0, CONTINUE, UNKNOWN]:
                midi_obj.markers.append(
                    Marker(text=str(token['chord']), time=curr_pos))

        elif token['type'] == 'Note':
            try:
                notes_list = all_notes[track2index[token['track']]]
            except:
                print(token)
                continue

            try:
                pitch = int(token['pitch'])
            except:
                pitch = pretty_midi.note_name_to_number(token['pitch'])

            duration = int(token['duration'])
            velocity = velocity_dict[token['track']]

            if int(duration) == 0:
                duration = 60
            end = curr_pos + int(duration)

            notes_list.append(
                Note(
                    pitch=pitch,
                    start=curr_pos,
                    end=end,
                    velocity=velocity)
            )

        else:
            print('ERROR! predicted type: ', token['type'])
            exit()

    # save midi
    melody_track = Instrument(0, is_drum=False, name='MELODY')
    melody_track.notes = all_notes[track2index['MELODY']]

    bridge_track = Instrument(0, is_drum=False, name='BRIDGE')
    bridge_track.notes = all_notes[track2index['BRIDGE']]

    piano_track = Instrument(0, is_drum=False, name='PIANO')
    piano_track.notes = all_notes[track2index['PIANO']]

    midi_obj.instruments = [melody_track, bridge_track, piano_track]
    midi_obj.dump(path_outfile)


if __name__ == '__main__':
    print('='*30)
    print('Start time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    args = parser.parse_args()

    # Save
    # if os.path.exists(args.save):
    #     os.system('rm -r {}'.format(args.save))
    if not os.path.exists(args.save):
        os.mkdir(args.save)

    # Device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.training:
        train()
    else:
        if not args.sampling_strategy_is_nucleus:
            for event in SAMPLING_HYPARAMETER:
                SAMPLING_HYPARAMETER[event] = {'t': 1, 'p': None}

        generate(num=args.max_generated_song_num, prompt_file=args.prompt_file)
