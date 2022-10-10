import os
import muspy
import pretty_midi


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


def get_structure_labels(song_id):
    structure_dir = '../../MusicData/hierarchical-structure-analysis/POP909/{:03d}'.format(
        song_id)
    with open(os.path.join(structure_dir, 'human_label1.txt'), 'r') as f:
        label1 = f.readlines()[0].strip()
    with open(os.path.join(structure_dir, 'human_label2.txt'), 'r') as f:
        label2 = f.readlines()[0].strip()

    sections_1 = get_pretty_structure(label1)
    sections_2 = get_pretty_structure(label2)

    return sections_1, sections_2


def read_chords(chord_file, pmidi):
    with open(chord_file, 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]

    chords = [l.split('\t') for l in lines]
    for c in chords:
        c[0] = pmidi.time_to_tick(float(c[0]))
        c[1] = pmidi.time_to_tick(float(c[1]))

    return chords


def read_midi(song_id, time_signature='4/4'):
    song_dir = '../../MusicData/POP909-Dataset/POP909/{:03d}'.format(song_id)
    song_file = os.path.join(song_dir, '{:03d}_aligned.mid'.format(song_id))
    chord_file = os.path.join(song_dir, 'chord_midi.txt')

    midi = muspy.read_midi(song_file)
    pmidi = pretty_midi.PrettyMIDI(song_file)
    chords = read_chords(chord_file, pmidi)

    if time_signature == '4/4':
        bar_window = 4 * midi.resolution
        return midi, bar_window
    else:
        print('Error')
        return None


def construct_a_new_midi(midi, start, end, header_name):
    new_midi = midi.deepcopy()
    for i, track in enumerate(midi):
        new_track_notes = [note for note in track if note.time >=
                           start and note.time <= end]
        # offset
        for note in new_track_notes:
            note.time -= start

        # print(i, len(new_midi), len(midi))
        new_midi[i].notes = new_track_notes
        new_midi[i].name += '_{}'.format(header_name)

    return new_midi


def partition_by_sections(song_id):
    label_1, label_2 = get_structure_labels(song_id)
    midi, bar_window = read_midi(song_id)

    save_dir = './partition/section/{:03d}'.format(song_id)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    def _partition(structure_label, annotation_index):
        partition_midis = []
        section_names = [' '.join([p['phrase'] for p in section])
                         for section in structure_label]

        for i, section in enumerate(structure_label):
            start = (section[0]['from'] - 1) * bar_window
            end = section[-1]['to'] * bar_window

            section_midi = construct_a_new_midi(
                midi, start, end, header_name=section_names[i])
            partition_midis.append(section_midi)

        for i, partition in enumerate(partition_midis):
            partition.write_midi(os.path.join(save_dir, 'label{}_{:03d}_section{}_{}.mid'.format(
                annotation_index, song_id, i+1, section_names[i])))

    _partition(label_1, 1)
    # _partition(label_2, 2)


def partition_by_phrase(song_id):
    label_1, label_2 = get_structure_labels(song_id)
    midi, bar_window = read_midi(song_id)

    save_dir = './partition/phrase/{:03d}'.format(song_id)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    def _partition(structure_label, annotation_index):
        partition_midis = []

        phrases = [p for section in structure_label for p in section]
        phrases_names = [p['phrase'] for p in phrases]

        for i, phrase in enumerate(phrases):
            start = (phrase['from'] - 1) * bar_window
            end = phrase['to'] * bar_window

            phrase_midi = construct_a_new_midi(
                midi, start, end, header_name=phrases_names[i])
            partition_midis.append(phrase_midi)

        for i, partition in enumerate(partition_midis):
            partition.write_midi(os.path.join(save_dir, 'label{}_{:03d}_phrase{}_{}.mid'.format(
                annotation_index, song_id, i+1, phrases_names[i])))

    _partition(label_1, 1)
    # _partition(label_2, 2)
