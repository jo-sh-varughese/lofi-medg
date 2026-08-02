import argparse
import json
import os
import random
import sys
from multiprocessing import Pool, cpu_count

import nibabel as nib
import nibabel.orientations as nio
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from lofi_utils.misc import convert_pad_space
from tools.preprocess_utils import resize_with_padding, save_jpg, seg_slice_to_boxes

'''
uv pip install nibabel
'''

LABEL_MAP = {
    1: "esophagus",
    2: "heart",
    3: "trachea",
    4: "aorta",
}
SEED = 0
DEFAULT_WINDOW_LEVEL = 40.0
DEFAULT_WINDOW_WIDTH = 400.0


def window_to_uint8(image, wl=DEFAULT_WINDOW_LEVEL, ww=DEFAULT_WINDOW_WIDTH):
    low = wl - (ww / 2.0)
    high = wl + (ww / 2.0)
    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    return (image * 255.0).astype(np.uint8)


def preprocess_case(
        case_id,
        volume_path,
        seg_path,
        split,
        save_dir,
        target_size,
        slice_stride,
        area_threshold,
):
    save_dict = {}

    volume_nii = nib.load(volume_path)
    seg_nii = nib.load(seg_path)

    volume = np.asarray(volume_nii.dataobj, dtype=np.float32)
    seg = np.asarray(seg_nii.dataobj, dtype=np.uint8)
    if volume.shape != seg.shape:
        return save_dict

    # canonicalize orientation so anatomical superior/inferior direction is stable
    # across patients; axial export below keeps spine on lower image side consistently.
    transform = nio.ornt_transform(nio.io_orientation(volume_nii.affine), nio.axcodes2ornt(("R", "A", "S")))
    volume = nio.apply_orientation(volume, transform)
    seg = nio.apply_orientation(seg, transform)

    for z in range(0, volume.shape[2], slice_stride):  # axial
        image = np.flipud(volume[:, :, z].T)
        image = window_to_uint8(image, wl=DEFAULT_WINDOW_LEVEL, ww=DEFAULT_WINDOW_WIDTH)

        seg_slice = np.flipud(seg[:, :, z].T)
        if not np.any(seg_slice):
            continue

        pil_image = Image.fromarray(image).convert("RGB")
        width, height = pil_image.size

        boxes_list = []
        for seg_id, label in LABEL_MAP.items():
            mask = (seg_slice == seg_id).astype(np.uint8)
            boxes = seg_slice_to_boxes(mask, area_threshold=area_threshold)
            if len(boxes) == 0:
                continue
            boxes = convert_pad_space(boxes, (width, height), target_size)
            boxes = [[int(v * target_size) for v in box] for box in boxes]
            boxes_list.append({"box": boxes, "label": label})

        if len(boxes_list) == 0:
            continue

        filename = f"segthor_{case_id}_{z:04d}.jpg"
        image_out_path = os.path.join(save_dir, filename)
        save_jpg(resize_with_padding(pil_image, target_size), image_out_path)
        save_dict[f"{split}/{filename}"] = boxes_list

    return save_dict


def split_cases(all_pairs, seed=0, n_val=4, n_test=4):
    rng = random.Random(seed)
    all_pairs = sorted(list(all_pairs), key=lambda x: x[0])
    rng.shuffle(all_pairs)

    val_pairs = all_pairs[:n_val]
    test_pairs = all_pairs[n_val: n_val + n_test]
    train_pairs = all_pairs[n_val + n_test:]
    return train_pairs, val_pairs, test_pairs


def find_case_pairs(segthor_train_dir):
    pairs = []
    for case_id in sorted(os.listdir(segthor_train_dir)):
        case_dir = os.path.join(segthor_train_dir, case_id)
        if not os.path.isdir(case_dir):
            continue
        if not case_id.startswith("Patient_"):
            continue

        volume_path = os.path.join(case_dir, f"{case_id}.nii.gz")
        seg_path = os.path.join(case_dir, "GT.nii.gz")
        if os.path.isfile(volume_path) and os.path.isfile(seg_path):
            pairs.append((case_id, volume_path, seg_path))
    return pairs


def run(args):
    all_pairs = find_case_pairs(args.segthor_dir)
    train_pairs, val_pairs, test_pairs = split_cases(all_pairs, seed=SEED, n_val=4, n_test=4)

    os.makedirs(args.output_dir, exist_ok=True)
    split_to_pairs = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
    }

    for split in split_to_pairs.keys():
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)

    split_json = {
        "train": {"modality": "CT", "data": {}},
        "val": {"modality": "CT", "data": {}},
        "test": {"modality": "CT", "data": {}},
    }

    for split, pairs in split_to_pairs.items():
        save_dir = os.path.join(args.output_dir, split)
        tasks = [
            (
                case_id,
                volume_path,
                seg_path,
                split,
                save_dir,
                args.target_size,
                args.slice_stride,
                args.area_threshold,
            )
            for case_id, volume_path, seg_path in pairs
        ]

        if args.num_workers <= 1 or len(tasks) <= 1:
            results = [
                (task[0], preprocess_case(*task))
                for task in tqdm(tasks, desc=f"preprocess {split}")
            ]
        else:
            with Pool(processes=min(args.num_workers, len(tasks))) as pool:
                save_dicts = pool.starmap(preprocess_case, tasks)
            results = list(zip((task[0] for task in tasks), save_dicts))

        for _, save_dict in sorted(results, key=lambda item: item[0]):
            split_json[split]["data"].update(save_dict)

    for split in ("train", "val", "test"):
        json_path = os.path.join(args.output_dir, f"{split}.json")
        with open(json_path, "wt", encoding="utf-8") as f:
            json.dump(split_json[split], f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segthor_dir",
        type=str,
    )
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--target_size", type=int)
    parser.add_argument("--slice_stride", type=int)
    parser.add_argument("--area_threshold", type=float)
    parser.add_argument("--num_workers", type=int)
    parser.set_defaults(
        segthor_dir="../data/SegTHOR/train/",
        output_dir="../data/segthor_512p/",
        target_size=512,
        slice_stride=2,
        area_threshold=0.01,
        num_workers=max(1, cpu_count() // 2),
    )
    args = parser.parse_args()
    run(args)
