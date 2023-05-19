import os
import shutil
import glob

import numpy as np
from tqdm import tqdm


input_path = "/ssd1/synthetic/synthetic_room_dataset"

output_path = "/ssd1/synthetic_0405"

room_dirs = ["rooms_04", "rooms_05"]


def create_dir(in_dir):
    """Creates directory if it does not exist"""
    if not os.path.isdir(in_dir):
        os.makedirs(in_dir)


create_dir(output_path)

for room_dir in room_dirs:
    print("DOING ROOM DIR: %s" % room_dir)
    room_path = os.path.join(input_path, room_dir)
    room_scenes = glob(os.path.join(room_path, "*/"))
    output_room_path = os.path.join(output_path, room_dir)

    non_overlap_scenes = []
    splits = {"test": [],
              "train": [],
              "val": []}
    for room_scene in tqdm(room_scenes):
        scene_info = os.path.join(room_scene, "item_dict.npz")
        item_dict = np.load(scene_info, allow_pickle=True)
        classes = item_dict["classes"]

        if len(np.unique(classes)) == int(item_dict["n_objects"]):
            print("FOUND ROOM SCENE: %s" % room_scene)
            non_overlap_scenes.append(room_scene)
            split = str(item_dict["split"])
            idx = "%08d" % item_dict['room_idx']
            splits[split].append(idx)
            shutil.copytree(room_scene, os.path.join(output_room_path, idx), symlinks=False, ignore=None)

    print("Found %d non-overlapping scenes" % len(non_overlap_scenes))
    for split, scene in splits.items():
        with open(os.path.join(output_room_path, split + ".lst"), "w") as f:
            f.write("\n".join(scene))
