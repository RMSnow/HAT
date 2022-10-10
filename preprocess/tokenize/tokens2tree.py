def delete_performance_tokens(tokens):
    # Key
    tokens = [t for t in tokens if t['type'] not in ['Key']]

    new_tokens = []
    has_tempo = False

    for t in tokens:
        # Velocity, Key
        for k in ['velocity', 'key']:
            if k in t:
                del t[k]

        # Tempo
        if t['type'] == 'Tempo':
            if not has_tempo:
                new_tokens.append(t)
                has_tempo = True
        else:
            new_tokens.append(t)

    return new_tokens


def refine_structure(tokens):
    onset = None
    onset_structure = None

    for i, t in enumerate(tokens):
        if onset is None and t['type'] in ['Chord', 'Note']:
            onset = i

        if onset_structure is None and t['type'] == 'Structure':
            onset_structure = i

        # i4 -> i
        if t['type'] == 'Structure':
            t['structure'] = t['structure'][0]

        # del <UNK>
        if t['structure'] == '<UNK>':
            t['structure'] = '<CONTI>'

    if onset_structure <= onset:
        return tokens

    # onset_structure > onset: 先有chord/note，后有structure
    new_tokens = []
    for i, t in enumerate(tokens):
        if i == onset:
            note_token = tokens[onset]
            structure_token = tokens[onset_structure]

            if structure_token['bar'] != '<CONTI>':
                tokens[onset_structure + 1]['bar'] = structure_token['bar']

            structure_token['bar'] = note_token['bar']
            structure_token['pos'] = note_token['pos']
            note_token['bar'] = '<CONTI>'

            new_tokens.append(structure_token)
            new_tokens.append(note_token)

            continue

        if i == onset_structure:
            continue

        new_tokens.append(t)

    return new_tokens


def get_sid(events, tokens_dict):
    for sid, items in tokens_dict.items():
        if events == items:
            return sid


def create_a_chord_token(tokens, start_index):
    curr_note = tokens[start_index]
    try:
        assert curr_note['type'] == 'Note' and curr_note['chord'] == '<CONTI>'
    except:
        print('Sid: {}, Type Error, {}, {}'.format(
            get_sid(tokens), start_index, curr_note))

    new_chord = {'type': 'Chord', 'bar': '<CONTI>', 'pos': curr_note['pos'], 'time': curr_note['time'],
                 'tempo': '<CONTI>', 'structure': '<CONTI>', 'chord': 0,
                 'track': 0, 'pitch': 0, 'duration': 0, 'velocity': 0}

    curr = start_index - 2
    last_chord = None
    while not last_chord:
        curr_token = tokens[curr]
        if curr_token['type'] == 'Chord':
            last_chord = curr_token['chord']
        else:
            try:
                assert curr_token['chord'] == '<CONTI>' or curr_token['type'] == 'Structure'
            except:
                print('Sid:{}, chord-conti Error, start_index: {}, curr: {}, {}'.format(
                    get_sid(tokens), start_index, curr, curr_token))

            curr -= 1

    new_chord['chord'] = last_chord
    return new_chord


def parse_chord(tokens, start_index, created_chord=None):
    if created_chord:
        curr_chord = created_chord
        curr = start_index
    else:
        curr_chord = tokens[start_index]
        curr = start_index + 1

    assert curr_chord['type'] == 'Chord'
    chord_tokens = [curr_chord]

    sz = len(tokens)

    while curr < sz:
        curr_token = tokens[curr]

        if curr_token['type'] != 'Note':
            break

        else:
            if curr_token['chord'] == '<CONTI>':
                chord_tokens.append(curr_token)
                curr += 1
            else:
                break

    return chord_tokens, curr


def parse_structure(tokens, start_index):
    curr_structure = tokens[start_index]
    assert curr_structure['type'] == 'Structure'
    structure_tokens = [curr_structure]

    sz = len(tokens)
    curr = start_index + 1

    new_created_chord_num = 0

    while curr < sz:
        curr_token = tokens[curr]

        if curr_token['type'] not in ['Chord', 'Note']:
            break

        elif curr_token['type'] == 'Chord':
            if curr_token['structure'] == '<CONTI>':
                chord_tokens, curr = parse_chord(tokens, curr)
                structure_tokens.append(chord_tokens)
            else:
                break

        else:
            if curr_token['structure'] == '<CONTI>':
                if curr_token['chord'] == '<CONTI>':
                    # Structure之中有一个单独Note，此时新建一个chord token，它的和弦应和之前保持一致
                    chord_tokens, curr = parse_chord(
                        tokens, curr, created_chord=create_a_chord_token(tokens, curr))
                    structure_tokens.append(chord_tokens)
                    new_created_chord_num += 1
                else:
                    # 单独一个没有chord约束的音符
                    structure_tokens.append(curr_token)
                    curr += 1
            else:
                break

    return structure_tokens, curr, new_created_chord_num


def parse_flatten_tokens(tokens):
    tree_tokens = []

    sz = len(tokens)
    curr = 0

    new_created_num = 0

    while curr < sz:
        curr_token = tokens[curr]

        if curr_token['type'] not in ['Structure', 'Chord']:
            tree_tokens.append(curr_token)
            curr += 1

        elif curr_token['type'] == 'Structure':
            structure_tokens, curr, new_created_chord_num = parse_structure(
                tokens, curr)
            tree_tokens.append(structure_tokens)
            new_created_num += new_created_chord_num

        else:
            chord_tokens, curr = parse_chord(tokens, curr)
            tree_tokens.append(chord_tokens)

    return tree_tokens, new_created_num


def parse_simul_notes(tokens):
    assert tokens[0]['type'] == 'Chord'

    tree_tokens = []
    sz = len(tokens)

    if sz == 1:
        return tokens

    curr_token = tokens[1]
    assert curr_token['type'] == 'Note'

    pos = curr_token['pos']
    simul_notes = [curr_token]

    curr = 2
    while curr < sz:
        curr_token = tokens[curr]
        assert curr_token['type'] == 'Note'

        if curr_token['bar'] == '<CONTI>' and curr_token['pos'] == pos:
            simul_notes.append(curr_token)
        else:
            tree_tokens.append(simul_notes)

            pos = curr_token['pos']
            simul_notes = [curr_token]

        curr += 1

    if len(simul_notes) != 0:
        tree_tokens.append(simul_notes)

    assert len([n for t in tree_tokens for n in t]) == len(tokens) - 1
    return [tokens[0]] + tree_tokens
