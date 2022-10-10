import numpy as np
import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note

BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4

MIN_VELOCITY = 40
DEFAULT_VELOCITY_BINS = np.linspace(0,  128, 64+1, dtype=np.int)

EVENT = {'type': 0, 'bar': 0, 'pos': 0, 'time': 0, 'tempo': 0, 'key': 0, 'structure': 0,
         'chord': 0, 'track': 0, 'pitch': 0, 'duration': 0, 'velocity': 0}

CONTINUE = '<CONTI>'
UNKNOWN = '<UNK>'


def create_sos_event():
    """
    Start
    """
    sos_event = EVENT.copy()
    sos_event['type'] = 'Boundary'
    sos_event['bar'] = '<SOS>'
    return sos_event


def create_eos_event():
    """
    End
    """
    eos_event = EVENT.copy()
    eos_event['type'] = 'Boundary'
    eos_event['bar'] = '<EOS>'
    return eos_event


def create_tempo_event(bar, pos, time, tempo):
    """
    Tempo change
    """
    tempo_event = EVENT.copy()
    tempo_event['type'] = 'Tempo'
    tempo_event['bar'] = bar
    tempo_event['pos'] = pos
    tempo_event['time'] = time
    tempo_event['tempo'] = tempo
    return tempo_event


def create_key_event(bar, pos, time, key):
    """
    Key change
    """
    key_event = EVENT.copy()
    key_event['type'] = 'Key'
    key_event['bar'] = bar
    key_event['pos'] = pos
    key_event['time'] = time
    key_event['key'] = key
    return key_event


def create_structure_event(bar, pos, time, structure):
    """
    Structure change
    """
    structure_event = EVENT.copy()
    structure_event['type'] = 'Structure'
    structure_event['bar'] = bar
    structure_event['pos'] = pos
    structure_event['time'] = time
    structure_event['structure'] = structure
    return structure_event


def create_chord_event(bar, pos, time, tempo, key, structure, chord):
    """
    Chord Change
    """
    chord_event = EVENT.copy()
    chord_event['type'] = 'Chord'
    chord_event['bar'] = bar
    chord_event['pos'] = pos
    chord_event['time'] = time
    chord_event['tempo'] = tempo
    chord_event['key'] = key
    chord_event['structure'] = structure
    chord_event['chord'] = chord
    return chord_event


def create_note_event(bar, pos, time, tempo, key, structure, chord, track, pitch, duration, velocity):
    """
    Note Change
    """
    note_event = EVENT.copy()
    note_event['type'] = 'Note'
    note_event['bar'] = bar
    note_event['pos'] = pos
    note_event['time'] = time
    note_event['tempo'] = tempo
    note_event['key'] = key
    note_event['structure'] = structure
    note_event['chord'] = chord
    note_event['track'] = track
    note_event['pitch'] = pitch
    note_event['duration'] = duration
    note_event['velocity'] = velocity
    return note_event


# def create_rest_event(bar):
#     rest_event = EVENT.copy()
#     rest_event['type'] = 'Rest_Bar'
#     rest_event['bar'] = bar
#     return rest_event


def get_event_tokens(data):
    global_end = data['metadata']['num_of_bars'] * BAR_RESOL

    final_sequence = [create_sos_event()]

    last_key_end_time = None
    last_structure_end_time = None
    last_chord_end_time = None

    rest_bar_cnt = 0

    def _get_bar_text(bar_events_seq, rest_bar_num):
        if len(bar_events_seq) == 0:
            bar_text = rest_bar_num + 1
        else:
            bar_text = CONTINUE

        rest_bar_num = 0
        return bar_text, rest_bar_num

    for bar_step in range(0, global_end, BAR_RESOL):
        bar_sequence = []

        for timing in range(bar_step, bar_step + BAR_RESOL, TICK_RESOL):
            pos_text = 'Beat_' + str((timing-bar_step)//TICK_RESOL)

            # Tempo
            t_tempos = data['tempos'][timing]
            if len(t_tempos) != 0:
                bar_text, rest_bar_cnt = _get_bar_text(
                    bar_sequence, rest_bar_cnt)

                tempo_text = int(t_tempos[-1].qpm)
                # time: mapped globally
                tempo_time = t_tempos[-1].time / global_end
                bar_sequence.append(create_tempo_event(
                    bar=bar_text, pos=pos_text, time=tempo_time, tempo=tempo_text))
            tempo_text = CONTINUE

            # Key
            t_keys = data['keys'][timing]
            if len(t_keys) != 0:
                bar_text, rest_bar_cnt = _get_bar_text(
                    bar_sequence, rest_bar_cnt)

                key_text = t_keys[-1]['name']
                # time: mapped globally
                key_time = t_keys[-1]['time'] / global_end
                key_duration = int(
                    np.round(t_keys[-1]['duration'] / TICK_RESOL) * TICK_RESOL)
                bar_sequence.append(create_key_event(
                    bar=bar_text, pos=pos_text, time=key_time, key=key_text))

                key_text = CONTINUE
                last_key_end_time = t_keys[-1]['time'] + key_duration
            else:
                key_text = CONTINUE if (
                    last_key_end_time and timing <= last_key_end_time) else UNKNOWN

            # Structure
            t_phrases = data['structure'][timing]
            if len(t_phrases) > 0:
                bar_text, rest_bar_cnt = _get_bar_text(
                    bar_sequence, rest_bar_cnt)

                structure_text = t_phrases[-1]['phrase']
                # time: mapped globally
                structure_time = t_phrases[-1]['time'] / global_end
                structure_duraion = t_phrases[-1]['duration'] * BAR_RESOL
                bar_sequence.append(create_structure_event(
                    bar=bar_text, pos=pos_text, time=structure_time, structure=structure_text))

                structure_text = CONTINUE
                last_structure_end_time = t_phrases[-1]['time'] + \
                    structure_duraion
            else:
                structure_text = CONTINUE if (
                    last_structure_end_time and timing <= last_structure_end_time) else UNKNOWN

            # Chord
            t_chords = data['chords'][timing]
            if len(t_chords) > 0:
                bar_text, rest_bar_cnt = _get_bar_text(
                    bar_sequence, rest_bar_cnt)

                chord_text = t_chords[-1]['name']
                # time: mapped locally
                chord_time = (t_chords[-1]['time'] - timing) / TICK_RESOL
                chord_duration = int(
                    np.round(t_chords[-1]['duration'] / TICK_RESOL) * TICK_RESOL)
                bar_sequence.append(create_chord_event(bar=bar_text, pos=pos_text, time=chord_time,
                                                       tempo=tempo_text, key=key_text, structure=structure_text, chord=chord_text))

                chord_text = CONTINUE
                last_chord_end_time = t_chords[-1]['time'] + chord_duration
            else:
                chord_text = CONTINUE if (
                    last_chord_end_time and timing <= last_chord_end_time) else UNKNOWN

            # Notes
            notes_events = []
            t_notes = []
            for track_name in ['MELODY', 'BRIDGE', 'PIANO']:
                for n in data['notes'][track_name][timing]:
                    t_notes.append({'track': track_name, 'note': n})

            t_notes = sorted(t_notes, key=lambda x: (
                x['note'].time, x['track'], x['note'].pitch))

            for note in t_notes:
                track_name = note['track']
                note = note['note']

                bar_text, rest_bar_cnt = _get_bar_text(
                    bar_sequence + notes_events, rest_bar_cnt)

                # time: mapped locally
                note_time = (note.time - timing) / TICK_RESOL

                # Duration: mapped
                ntick_duration = int(
                    np.round(note.duration / TICK_RESOL) * TICK_RESOL)
                note.duration = ntick_duration

                # Velocity: mapped
                note.velocity = DEFAULT_VELOCITY_BINS[np.argmin(
                    abs(DEFAULT_VELOCITY_BINS-note.velocity))]
                note.velocity = max(MIN_VELOCITY, note.velocity)

                notes_events.append(create_note_event(bar=bar_text, pos=pos_text, time=note_time, tempo=tempo_text, key=key_text,
                                                      structure=structure_text, chord=chord_text, track=track_name, pitch=note.pitch, duration=note.duration, velocity=note.velocity))

            # notes_events = sorted(
            #     notes_events, key=lambda x: (x['time'], x['track'], x['pitch']))
            bar_sequence += notes_events

        if len(bar_sequence) == 0:
            # bar_sequence.append(create_rest_event(bar='Bar'))
            rest_bar_cnt += 1

        final_sequence += bar_sequence

    # EOS
    final_sequence.append(create_eos_event())

    return final_sequence


def write_midi(words, path_outfile, write_chord=False, write_key=False, write_structure=True):
    midi_obj = miditoolkit.midi.parser.MidiFile()

    all_notes = [[], [], []]
    track2index = {'MELODY': 0, 'BRIDGE': 1, 'PIANO': 2}

    bar_cnt = -1

    for i in range(len(words)):
        token = words[i]

        if token['bar'] in ['<SOS>', '<EOS>']:
            continue

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

        elif token['type'] == 'Key':
            if write_key and token['key'] not in [0, CONTINUE, UNKNOWN]:
                midi_obj.markers.append(
                    Marker(text=str(token['key']), time=curr_pos))

        elif token['type'] == 'Structure':
            if write_structure and token['structure'] not in [0, CONTINUE, UNKNOWN]:
                midi_obj.markers.append(
                    Marker(text=token['structure'], time=curr_pos))

        elif token['type'] == 'Chord':
            if write_chord and token['chord'] not in [0, CONTINUE, UNKNOWN]:
                midi_obj.markers.append(
                    Marker(text=str(token['chord']), time=curr_pos))

        elif token['type'] == 'Note':
            notes_list = all_notes[track2index[token['track']]]

            pitch = int(token['pitch'])
            duration = int(token['duration'])
            velocity = int(token['velocity'])

            if int(duration) == 0:
                duration = 60
            end = curr_pos + int(duration)

            notes_list.append(
                Note(
                    pitch=int(pitch),
                    start=curr_pos,
                    end=end,
                    velocity=int(velocity))
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
