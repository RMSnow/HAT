import sys
import os
from tqdm import tqdm

if __name__ == '__main__':
    midi_dir = '/data/zhangxueyao/WCPR/MusicData/POP909-Dataset/POP909'
    midi_files = []
    for root, _, files in os.walk(midi_dir):
        for f in files:
            if '.mid' not in f:
                continue
            midi_files.append(os.path.join(root, f))

    for input_file in tqdm(midi_files):
        try:
            output_file = input_file.replace('.mid', '.mp3')
            os.system(
                'timidity {} -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 64k {}'.format(input_file, output_file))
        except:
            continue
