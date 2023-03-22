import os
from miditoolkit import midi
from tqdm import tqdm
from get_texture_stability import calculate_texture_stability_of_cp_corpus
import pretty_midi
import json
import pandas as pd

max_pitch = pretty_midi.note_name_to_number('C4')

del_songs = []
del_songs.append('MusicTransformer-00003')

if __name__ == '__main__':
    # === Calculate Texture Stability === #
    grooves_save_dir = 'grooves'

    generated_data_dir = '/data/zhangxueyao/WCPR/StructureMultiTrack/model/TextureAndForm/主观评价/midi_raw'
    cp_corpus_dir = '/data/zhangxueyao/WCPR/StructureMultiTrack/model/TextureAndForm/客观评价/texture/cp/corpus'

    for midi_dir in os.listdir(generated_data_dir):
        if midi_dir == 'Real':
            continue

        model_file = os.path.join(grooves_save_dir, '{}.json'.format(midi_dir))
        if not os.path.exists(model_file):
            print('\nEval {}...\n'.format(midi_dir))

            model_dict = dict()
            for midi_file in tqdm(os.listdir(os.path.join(generated_data_dir, midi_dir))):
                if '.mid' not in midi_file:
                    continue

                cp_file = os.path.join(
                    cp_corpus_dir, midi_dir, midi_file + '.pkl')
                bar_level_res, chord_level_res = calculate_texture_stability_of_cp_corpus(
                    cp_file, max_pitch)

                model_dict[midi_file] = {
                    'bar': bar_level_res, 'chord': chord_level_res}

            with open(model_file, 'w') as f:
                json.dump(model_dict, f, indent=4, ensure_ascii=False)

    #  === Every Model's Result ===
    report_dir = 'report'
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)

    model_report_file = os.path.join(report_dir, 'models.json')
    res_dict = dict()
    for file in tqdm(os.listdir(grooves_save_dir)):
        model = file.split('.')[0]
        res_dict[model] = dict()

        with open(os.path.join(grooves_save_dir, file), 'r') as f:
            model_dict = json.load(f)

        grams_ranges = list(model_dict.values())[0].keys()
        for key in grams_ranges:
            values = [item[key] for song, item in model_dict.items(
            ) if song.replace('.mid', '') not in del_songs]

            if key == 'chord' and model != 'Real':
                values.sort()
                print(model)
                for v in values:
                    print(v)
                print()

            df = pd.DataFrame({'val': values})
            mean = df['val'].mean()
            std = df['val'].std()

            res_dict[model][key] = {'mean': mean, 'std': std}

    with open(model_report_file, 'w') as f:
        json.dump(res_dict, f, indent=4, ensure_ascii=False)

    # #  === Final report ===
    # report_file = os.path.join(report_dir, 'report.json')
    # with open(model_report_file, 'r') as f:
    #     model_report_dict = json.load(f)

    # report_dict = dict()
    # real_data_dict = model_report_dict['Real']
    # for key in real_data_dict.keys():
    #     report_dict[key] = dict()
    #     real_mean, real_std = real_data_dict[key]['mean'], real_data_dict[key]['std']

    #     for model_name, item in model_report_dict.items():
    #         mean, std = item[key]['mean'], item[key]['std']
    #         distance = (mean - real_mean) ** 2 + \
    #             abs(std + real_std - 2 * (std * real_std)**0.5)

    #         report_dict[key][model_name] = distance

    # with open(report_file, 'w') as f:
    #     json.dump(report_dict, f, indent=4, ensure_ascii=False)
