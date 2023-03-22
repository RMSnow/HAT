from tqdm import tqdm
import pretty_midi
import os
import json
from get_texture_stability import calculate_texture_stability_of_cp_corpus

cp_raw_dir = '/data/zhangxueyao/WCPR/StructureMultiTrack/model/TextureAndForm/客观评价/texture/cp-real/corpus'


eval_dict = dict()
for file in tqdm(os.listdir(cp_raw_dir)):
    filename = file.split('.')[0]
    file = os.path.join(cp_raw_dir, file)
    bar_level_res, chord_level_res = calculate_texture_stability_of_cp_corpus(
        file, max_pitch=pretty_midi.note_name_to_number('C4'))

    eval_dict[filename] = dict()

    eval_dict[filename]['bar'] = bar_level_res
    eval_dict[filename]['chord'] = chord_level_res


eval_file = os.path.join('grooves/Real.json')
with open(eval_file, 'w') as f:
    json.dump(eval_dict, f, indent=4, ensure_ascii=False)