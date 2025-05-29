import librosa
import numpy as np
import os
import shutil
from PIL import Image

def preprocess(mp3_path, temp_output_dir):
    sample_rate = 22050
    chunk_duration = 5
    chunk_samples = sample_rate * chunk_duration

    y, sr = librosa.load(mp3_path, sr=sample_rate)
    total_samples = len(y)
    num_chunks = total_samples // chunk_samples
    num_full_groups = num_chunks // 4

    group_dirs = []

    for group_idx in range(num_full_groups):
        group_dir = os.path.join(temp_output_dir, f"group_{group_idx}")
        os.makedirs(group_dir, exist_ok=True)
        group_dirs.append(group_dir)

        for i in range(4):
            chunk_idx = group_idx * 4 + i
            start = chunk_idx * chunk_samples
            end = start + chunk_samples
            chunk = y[start:end]

            S = librosa.stft(chunk)
            S_db = librosa.amplitude_to_db(np.abs(S))

            S_img = Image.fromarray(S_db)
            S_img = S_img.resize((128, 128))
            S_arr = np.array(S_img)

            S_norm = (S_arr - S_arr.min()) / (S_arr.max() - S_arr.min() + 1e-6)
            np.save(os.path.join(group_dir, f"{i}.npy"), S_norm)

    print(f"Сохранено {num_full_groups} групп из {mp3_path}")
    return group_dirs

def move_groups(groups, destination_base, part, label_source_name, start_index):
    part_dir = os.path.join(destination_base, part)
    os.makedirs(part_dir, exist_ok=True)

    label = "chainsaw" if "chainsaw" in label_source_name else "other"

    for i, group in enumerate(groups):
        group_name = f"group_{start_index + i}"
        dst = os.path.join(part_dir, group_name)
        shutil.move(group, dst)

        label_file_path = os.path.join(dst, "label.txt")
        with open(label_file_path, "w") as f:
            f.write(label)

    return start_index + len(groups)


mp3_paths = ["data/chainsaw1.mp3", "data/chainsaw2.mp3", "data/chainsaw3.mp3", "data/chainsaw4.mp3", "data/forest_relax.mp3", "data/street.mp3", "data/val_chainsaw.mp3"]
temp_dirs = ["temp/chainsaw1", "temp/chainsaw2", "temp/chainsaw3", "temp/chainsaw4", "temp/forest_relax", "temp/street", "temp/val_chainsaw"]
final_output_base = "dataset"


chainsaw1_groups = preprocess(mp3_paths[0], temp_dirs[0])
chainsaw2_groups = preprocess(mp3_paths[1], temp_dirs[1])
chainsaw3_groups = preprocess(mp3_paths[2], temp_dirs[2])
chainsaw4_groups = preprocess(mp3_paths[3], temp_dirs[3])
forest_groups = preprocess(mp3_paths[4], temp_dirs[4])
street_groups = preprocess(mp3_paths[5], temp_dirs[5])
val_chainsaw_groups = preprocess(mp3_paths[6], temp_dirs[6])

part1_index = 0
part2_index = 0

part1_index = move_groups(chainsaw1_groups, final_output_base, "train", "chainsaw1", part1_index)
part1_index = move_groups(chainsaw2_groups, final_output_base, "train", "chainsaw2", part1_index)


part2_index = move_groups(chainsaw3_groups, final_output_base, "val", "chainsaw3", part2_index)
part2_index = move_groups(chainsaw4_groups, final_output_base, "val", "chainsaw4", part2_index)


part1_index = move_groups(val_chainsaw_groups, final_output_base, "train", "val_chainsaw", part1_index)

train_forest = forest_groups[:int(len(forest_groups) * 0.6)]
val_forest = forest_groups[int(len(forest_groups) * 0.7):]

part1_index = move_groups(train_forest, final_output_base, "train", "forest_relax", part1_index)
part2_index = move_groups(val_forest, final_output_base, "val", "forest_relax", part2_index)

train_street = street_groups[:int(len(street_groups) * 0.6)]
val_street = street_groups[int(len(street_groups) * 0.7):]

part1_index = move_groups(train_street, final_output_base, "train", "street", part1_index)
part2_index = move_groups(val_street, final_output_base, "val", "street", part2_index)
