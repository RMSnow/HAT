from tqdm import tqdm
import random
import pretty_midi
import muspy
import mido
import pandas as pd
import os
import numpy as np
import collections
import pickle
import json


BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4


def read_chords(chord_file, pmidi, resolution):
    factor = BEAT_RESOL / resolution

    with open(chord_file, 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]

    chords = [l.split('\t') for l in lines]
    for c in chords:
        c[0] = pmidi.time_to_tick(float(c[0])) * factor
        c[1] = pmidi.time_to_tick(float(c[1])) * factor

        c[0] = round(c[0])
        c[1] = round(c[1])

    chords = [{'time': int(c[0]), 'duration':int(c[1] - c[0]), 'name':c[2]}
              for c in chords if c[2] != 'N']
    return chords


def read_keys(key_file, pmidi, resolution):
    factor = BEAT_RESOL / resolution

    with open(key_file, 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]

    keys = [l.split('\t') for l in lines]
    for k in keys:
        k[0] = pmidi.time_to_tick(float(k[0])) * factor
        k[1] = pmidi.time_to_tick(float(k[1])) * factor

        k[0] = round(k[0])
        k[1] = round(k[1])

    keys = [{'time': int(k[0]), 'duration':int(k[1] - k[0]), 'name':k[2]}
            for k in keys]
    return keys


def get_pretty_structure(phrases):
    sections = []

    curr = 0
    curr_sep = []
    curr_section = []

    while curr <= len(phrases) - 1:
        melody_type = phrases[curr]
        curr += 1

        measures = ''
        while curr <= len(phrases) - 1 and phrases[curr].isdigit():
            measures += phrases[curr]
            curr += 1
        measures = int(measures)

        word = '{}{}'.format(melody_type, measures)

        if melody_type.islower() or (melody_type == 'X' and measures > 2):
            if len(curr_section) != 0:
                sections.append(' '.join(curr_section))
                curr_section = []

            curr_sep.append(word)

        else:
            if len(curr_sep) != 0:
                sections.append(' '.join(curr_sep))
                curr_sep = []

            curr_section.append(word)

    if len(curr_section) == 0 and len(curr_sep) == 0:
        print('ERROR')
        return None

    if len(curr_section) != 0:
        sections.append(' '.join(curr_section))

    if len(curr_sep) != 0:
        sections.append(' '.join(curr_sep))

    raw_len = len(phrases)
    new_len = len(' '.join(sections).replace(' ', ''))
    assert raw_len == new_len

    return get_pretty_section(sections)


def get_pretty_section(sections):
    pretty_sections = []

    num_measures = 1
    for sec in sections:
        pretty_sec = []
        for word in sec.split():
            tmp_num = int(word[1:])
            phrase = {'phrase': word, 'from': num_measures, 'to': num_measures +
                      tmp_num - 1, 'name': word[0], 'duration': tmp_num}
            pretty_sec.append(phrase)
            num_measures += tmp_num

        pretty_sections.append(pretty_sec)

    return pretty_sections


def read_structure(structure_file):
    with open(structure_file, 'r') as f:
        label = f.readlines()[0].strip()
    return get_pretty_structure(label)


def read_midi(song_id):
    song_id = int(song_id)

    song_dir = '../../../MusicData/POP909-Dataset/POP909/{:03d}'.format(
        song_id)
    song_file = os.path.join(song_dir, '{:03d}.mid'.format(song_id))
    chord_file = os.path.join(song_dir, 'chord_midi.txt')
    key_file = os.path.join(song_dir, 'key_audio.txt')

    structure_dir = '../../../MusicData/hierarchical-structure-analysis/POP909/{:03d}'.format(
        song_id)
    structure_file1 = os.path.join(structure_dir, 'human_label1.txt')
    structure_file2 = os.path.join(structure_dir, 'human_label2.txt')

    midi = muspy.read_midi(song_file)
    pmidi = pretty_midi.PrettyMIDI(song_file)
    chords = read_chords(chord_file, pmidi, midi.resolution)
    keys = read_keys(key_file, pmidi, midi.resolution)

    # adjust resolution
    midi.adjust_resolution(BEAT_RESOL)

    # Structure
    structure_1 = read_structure(structure_file1)
    structure_2 = read_structure(structure_file2)

    return midi, chords, keys, (structure_1, structure_2)


def get_grouped_events(song_id, midi, chords, keys, structures, token_format='flatten'):
    first_note_time = min([n.time for track in midi for n in track.notes])
    if token_format != 'flatten':
        # 从piano track的第一个note开始算时间
        first_note_time = midi[-1].notes[0].time

    last_note_time = max([n.time for track in midi for n in track.notes])
    quant_time_first = int(np.round(first_note_time / TICK_RESOL) * TICK_RESOL)
    offset = quant_time_first // BAR_RESOL  # empty bar
    last_bar = int(np.ceil(last_note_time / BAR_RESOL)) - offset

    # Track / Notes
    tracks_groups = dict()
    for track in midi:
        notes_groups = collections.defaultdict(list)
        for note in track.notes:
            if note.time < first_note_time:
                continue

            # time delete offset
            note.time -= offset * BAR_RESOL

            # group idx
            quant_time = int(np.round(note.time / TICK_RESOL) * TICK_RESOL)
            notes_groups[quant_time].append(note)

        tracks_groups[track.name] = notes_groups

    # Tempo
    tempos_groups = collections.defaultdict(list)
    
    for tempo in midi.tempos:
        tempo.time -= offset * BAR_RESOL
        tempo.time = 0 if tempo.time < 0 else tempo.time

        quant_time = int(np.round(tempo.time / TICK_RESOL) * TICK_RESOL)
        tempos_groups[quant_time].append(tempo)

        if token_format != 'flatten':
            break

    # Chord
    chords_groups = collections.defaultdict(list)
    for chord in chords:
        if chord['time'] + chord['duration'] < first_note_time:
            continue

        chord['time'] -= offset * BAR_RESOL
        # chord['time'] = 0 if chord['time'] < 0 else chord['time']

        quant_time = int(np.round(chord['time'] / TICK_RESOL) * TICK_RESOL)
        chords_groups[quant_time].append(chord)

    # Key
    keys_groups = collections.defaultdict(list)

    if token_format != 'flatten':
        keys = []

    for key in keys:
        key['time'] -= offset * BAR_RESOL
        key['time'] = 0 if key['time'] < 0 else key['time']

        quant_time = int(np.round(key['time'] / TICK_RESOL) * TICK_RESOL)
        keys_groups[quant_time].append(key)

    # Structure
    """
    phrases: eg:
    [{'phrase': 'i4', 'from': 1, 'to': 4, 'name': 'i', 'duration': 4},
    {'phrase': 'A4', 'from': 5, 'to': 8, 'name': 'A', 'duration': 4},
    {'phrase': 'B8', 'from': 9, 'to': 16, 'name': 'B', 'duration': 8},
    {'phrase': 'A4', 'from': 17, 'to': 20, 'name': 'A', 'duration': 4},
    ...]
    """
    phrases = [p for sec in structures for p in sec]
    structures_groups = collections.defaultdict(list)

    # 从piano track的第一个note开始算时间
    structure_start_time  = first_note_time - offset * BAR_RESOL
    for phrase in phrases:
        phrase['time'] = structure_start_time + \
            (phrase['from'] - 1) * BAR_RESOL

        quant_time = int(np.round(phrase['time'] / TICK_RESOL) * TICK_RESOL)
        structures_groups[quant_time].append(phrase)

    song_data = {'notes': tracks_groups, 'chords': chords_groups, 'tempos': tempos_groups,
                 'keys': keys_groups, 'structure': structures_groups, 'metadata': {'song_id': song_id, 'offset_blank_bars': offset, 'num_of_bars': last_bar}}

    return song_data
