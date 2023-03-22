import pickle
from numpy.core.defchararray import mod
from tqdm import tqdm
import os
import miditoolkit
from chorder import Dechorder
import json
import pandas as pd
import numpy as np

invalid_chord = "N_N_N"

irregularity = "irregularity"
reasonability = "reasonability"

del_songs = []
del_songs.append("MusicTransformer-00003")


def parse_chords_of_file(midi_file):
    midi_obj = miditoolkit.midi.parser.MidiFile(midi_file)
    chords = Dechorder.dechord(midi_obj)

    num2pitch = {
        0: "C",
        1: "C#",
        2: "D",
        3: "D#",
        4: "E",
        5: "F",
        6: "F#",
        7: "G",
        8: "G#",
        9: "A",
        10: "A#",
        11: "B",
    }

    markers = []
    for cidx, chord in enumerate(chords):
        if chord.is_complete():
            chord_text = (
                num2pitch[chord.root_pc]
                + "_"
                + chord.quality
                + "_"
                + num2pitch[chord.bass_pc]
            )
            # chord_text = num2pitch[chord.root_pc]
        else:
            chord_text = invalid_chord

        markers.append(chord_text)

    return markers


def parse_chords(midi_dir, save_file):
    chords_dict = dict()

    print("正在解析{}中的和弦...".format(midi_dir))
    for file in tqdm(os.listdir(midi_dir)):
        if ".mid" not in file:
            continue

        chords = parse_chords_of_file(os.path.join(midi_dir, file))
        chords_dict[file] = chords

    with open(save_file, "w") as f:
        json.dump(chords_dict, f, indent=4, ensure_ascii=False)


def calculate_irregularity_of_sequences(chords, grams):
    sz = len(chords)

    unique_cp = set()
    total_cp = []

    for i in range(sz):
        if i + grams >= sz:
            break

        cp = tuple(chords[i : i + grams])
        if invalid_chord in cp:
            continue

        total_cp.append(cp)
        unique_cp.add(cp)

    res = len(unique_cp) / len(total_cp)
    return res


def calculate_reasonality_of_sequences(chords, grams):
    sz = len(chords)
    unique_cp = set()

    scores = []

    context = None
    for i in range(sz):
        if i + grams >= sz:
            break

        cp = tuple(chords[i : i + grams])
        if invalid_chord in cp:
            continue

        if context is None:
            # the first cp
            unique_cp.add(cp)
            context = chords[i : i + grams]
        elif cp in unique_cp:
            continue
        else:
            unique_cp.add(cp)

            # look up the context dict
            context_num = chords_context_dict[grams][0][tuple(context)]
            context_num = 1 if context_num == 0 else context_num
            num = chords_context_dict[grams][1][tuple(context + [cp[-1]])]
            scores.append(num / context_num)

            context = chords[i : i + grams]

    res = sum(scores) / len(scores)
    return res


if __name__ == "__main__":
    # === Parse Chords ===
    chords_save_dir = "chords"
    if not os.path.exists(chords_save_dir):
        os.mkdir(chords_save_dir)

    real_data_chords_file = os.path.join("chords", "Real.json")
    if not os.path.exists(real_data_chords_file):
        real_midi_dir = "/data/zhangxueyao/WCPR/MusicData/POP909-Dataset/only_midi"
        parse_chords(real_midi_dir, real_data_chords_file)

    generated_data_dir = (
        "/data/zhangxueyao/WCPR/StructureMultiTrack/model/TextureAndForm/主观评价/midi_raw"
    )
    sub_data_dirs = [
        "/data/zhangxueyao/WCPR/StructureMultiTrack/model/TextureAndForm/ckpt/BaseRelative/loss_5/Sub-MusicTransformer"
    ]
    sub_data_dirs.append(
        "/data/zhangxueyao/WCPR/StructureMultiTrack/model/TextureAndForm/ckpt/Base/loss_5/Sub-HAT-base"
    )
    sub_data_dirs.append(
        "/data/zhangxueyao/WCPR/StructureMultiTrack/model/TextureAndForm/ckpt/HAT_without_form/loss_5/Sub-HAT-base-with-texture"
    )
    sub_data_dirs.append(
        "/data/zhangxueyao/WCPR/StructureMultiTrack/model/TextureAndForm/ckpt/HAT_without_texture/loss_5/Sub-HAT-base-with-form"
    )

    for dir in os.listdir(generated_data_dir):
        if dir == "Real":
            continue

        chords_file = os.path.join("chords", "{}.json".format(dir))
        if not os.path.exists(chords_file):
            parse_chords(dir, chords_file)

    for dir in sub_data_dirs:
        dir_name = dir.split("/")[-1]
        chords_file = os.path.join("chords", "{}.json".format(dir_name))
        if not os.path.exists(chords_file):
            parse_chords(os.path.join(generated_data_dir, dir), chords_file)

    # === Calculate Irregularity === #
    irregularity_save_dir = "irregularity"
    if not os.path.exists(irregularity_save_dir):
        os.mkdir(irregularity_save_dir)

    for chords_file in os.listdir(chords_save_dir):
        model_name = chords_file.split(".")[0]
        save_file = os.path.join(irregularity_save_dir, "{}.json".format(model_name))

        if not os.path.exists(save_file):
            print("正在计算{}的Irregularity...".format(chords_file))

            with open(os.path.join(chords_save_dir, chords_file), "r") as f:
                chords_dict = json.load(f)

            cp_irregularity_dict = dict()
            for file, chords in tqdm(chords_dict.items()):
                cp_irregularity_dict[file] = dict()
                for N in [2, 3, 4]:
                    cp_irregularity_dict[file][
                        "{}-grams".format(N)
                    ] = calculate_irregularity_of_sequences(chords, N)

            with open(save_file, "w") as f:
                json.dump(cp_irregularity_dict, f, indent=4, ensure_ascii=False)

    # === Calculate Reasonability ===
    reasonability_save_dir = "reasonability"
    if not os.path.exists(reasonability_save_dir):
        os.mkdir(reasonability_save_dir)

    with open("./chords_context/chords_context_dict.pkl", "rb") as f:
        chords_context_dict = pickle.load(f)

    for chords_file in os.listdir(chords_save_dir):
        model_name = chords_file.split(".")[0]
        save_file = os.path.join(reasonability_save_dir, "{}.json".format(model_name))

        if not os.path.exists(save_file):
            print("正在计算{}Reasonability...".format(chords_file))

            with open(os.path.join(chords_save_dir, chords_file), "r") as f:
                chords_dict = json.load(f)

            cp_reasonability_dict = dict()
            for file, chords in tqdm(chords_dict.items()):
                cp_reasonability_dict[file] = dict()
                for N in [2, 3, 4]:
                    cp_reasonability_dict[file][
                        "{}-grams".format(N)
                    ] = calculate_reasonality_of_sequences(chords, N)

            with open(save_file, "w") as f:
                json.dump(cp_reasonability_dict, f, indent=4, ensure_ascii=False)

    #  === Final Report ===
    report_dir = "report"
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)

    # Every model's results
    for file in tqdm(os.listdir(irregularity_save_dir)):
        model_name = file.split(".")[0]
        model_save_file = os.path.join(report_dir, "{}.json".format(model_name))

        CPI_file = os.path.join(irregularity_save_dir, file)
        CPVR_file = os.path.join(reasonability_save_dir, file)

        with open(CPI_file, "r") as f:
            CPI_dict = json.load(f)
        with open(CPVR_file, "r") as f:
            CPVR_dict = json.load(f)

        grams_ranges = list(CPI_dict.values())[0].keys()
        model_res = dict()
        for midi in CPI_dict:
            model_res[midi] = dict()

            values = []
            for gram in grams_ranges:
                res = (1 - CPI_dict[midi][gram] + CPVR_dict[midi][gram]) / 2
                model_res[midi][gram] = res
                values.append(res)

            model_res[midi]["avg_all_grams"] = np.mean(values)

        # sort midis
        model_res = sorted(model_res.items(), key=lambda x: x[1]["avg_all_grams"])
        with open(model_save_file, "w") as f:
            json.dump(model_res, f, indent=4, ensure_ascii=False)

    # Final
    model_report_file = os.path.join(report_dir, "_report.json")
    res_dict = dict()
    for file in tqdm(os.listdir(irregularity_save_dir)):
        model = file.split(".")[0]
        res_dict[model] = dict()

        # irregularity_save_dir
        with open(os.path.join(irregularity_save_dir, file), "r") as f:
            model_dict = json.load(f)

        grams_ranges = list(model_dict.values())[0].keys()
        for key in grams_ranges:
            values = [
                item[key]
                for song, item in model_dict.items()
                if song.replace(".mid", "") not in del_songs
            ]

            df = pd.DataFrame({"val": values})
            mean = df["val"].mean()
            std = df["val"].std()

            res_dict[model][key] = {irregularity: {"mean": mean, "std": std}}

        # reasonability_save_dir
        with open(os.path.join(reasonability_save_dir, file), "r") as f:
            model_dict = json.load(f)

        grams_ranges = list(model_dict.values())[0].keys()
        for key in grams_ranges:
            values = [
                item[key]
                for song, item in model_dict.items()
                if song.replace(".mid", "") not in del_songs
            ]

            df = pd.DataFrame({"val": values})
            mean = df["val"].mean()
            std = df["val"].std()

            res_dict[model][key][reasonability] = {"mean": mean, "std": std}

        # 汇总
        for key in grams_ranges:
            a = 1 - res_dict[model][key][irregularity]["mean"]
            b = res_dict[model][key][reasonability]["mean"]
            res_dict[model][key]["Avg"] = (a + b) / 2

    with open(model_report_file, "w") as f:
        json.dump(res_dict, f, indent=4, ensure_ascii=False)
