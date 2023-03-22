import pickle
import pandas as pd
import random
from tqdm import tqdm
import numpy as np

BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4


def corpus2groups(cp_corpus_file):
    data = pickle.load(open(cp_corpus_file, 'rb'))

    # global tag
    global_end = data['metadata']['last_bar'] * BAR_RESOL

    # process
    final_sequence = []
    for bar_step in range(0, global_end, BAR_RESOL):
        curr_bar = []
        for timing in range(bar_step, bar_step + BAR_RESOL, TICK_RESOL):
            t_chords = data['chords'][timing]
            t_notes = data['notes'][0][timing]  # piano track

            t_chords = [{'chord': c.text} for c in t_chords]
            t_notes = [{'note': n.pitch} for n in t_notes]

            curr_bar.append(t_chords + t_notes)
        final_sequence.append(curr_bar)

    return final_sequence


def get_bar_level_texture(bars, max_pitch=-1):
    arr = np.zeros((len(bars), 16))

    for i, bar in enumerate(bars):
        for j, beat in enumerate(bar):
            if len(beat) == 0:
                continue

            notes = [event for event in beat if 'note' in event]
            if max_pitch != -1:
                notes = [n for n in notes if n['note'] <= max_pitch]

            if len(notes) > 0:
                arr[i, j] = 1

    return arr


def get_chord_level_texture(bars, max_pitch=-1):
    beats = [beat for bar in bars for beat in bar]

    chord_groups = []
    curr_chord = []
    for beat in beats:
        if len(beat) == 0:
            curr_chord.append(0)
            continue

        if 'chord' in beat[0] and len(curr_chord) != 0:
            chord_groups.append(curr_chord)
            curr_chord = []

        notes = [event for event in beat if 'note' in event]
        if max_pitch != -1:
            notes = [n for n in notes if n['note'] <= max_pitch]

        if len(notes) > 0:
            curr_chord.append(1)
        else:
            curr_chord.append(0)

    if len(curr_chord) != 0:
        chord_groups.append(curr_chord)

    return chord_groups


def calculate_texture_stability_of_cp_corpus(cp_corpus_file, max_pitch):
    bars = corpus2groups(cp_corpus_file)

    # === Bar-Level ===
    # (#bars, 16)
    bar_level_texture = get_bar_level_texture(bars, max_pitch)
    # (#bars-1, 16)
    arr1, arr2 = bar_level_texture[:-1], bar_level_texture[1:]

    # # (#bars-1, 16)
    # res = np.logical_xor(arr1, arr2)
    # bar_level_res = 1 - np.sum(res) / (arr1.shape[0] * arr1.shape[1])

    # 去掉全部是0的
    bar_level_res = []
    for i, a in enumerate(arr1):
        b = arr2[i]

        if np.sum(a) == 0 and np.sum(b) == 0:
            continue
        else:
            res = 1 - np.sum(np.logical_xor(a, b)) / \
                np.sum(np.logical_or(a, b))
            bar_level_res.append(res)

    bar_level_res = sum(bar_level_res) / len(bar_level_res)

    # === Chord-Level ===
    chord_level_texture = get_chord_level_texture(bars, max_pitch)
    sz = len(chord_level_texture)

    chord_level_res = []
    for i in range(sz-1):
        arr1, arr2 = chord_level_texture[i], chord_level_texture[i+1]

        if len(arr1) != len(arr2):
            continue
        elif np.sum(arr1) == 0 and np.sum(arr2) == 0:
            continue
        else:
            res = 1 - np.sum(np.logical_xor(arr1, arr2)) / \
                np.sum(np.logical_or(arr1, arr2))
            chord_level_res.append(res)

    chord_level_res = sum(chord_level_res) / len(chord_level_res)

    return bar_level_res, chord_level_res
