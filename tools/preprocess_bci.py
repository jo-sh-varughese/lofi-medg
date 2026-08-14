'''
Build the HER2 IHC track of the `her2` dataset from BCI.

BCI (Breast Cancer Immunohistochemical, Liu et al., CVPR-W 2022) is the only
public source that pairs *IHC pixels* with a *HER2 expression grade for exactly
those pixels*. Patches are named `<id>_<split>_<grade>.png` with the grade in
{0, 1+, 2+, 3+}, which is the real ground truth we use.

Supervision honesty
-------------------
    label text : REAL. Taken from the BCI grade in the filename and rendered
                 through a fixed template in her2/captions.py.
    box        : HEURISTIC. Connected components of the DAB (brown chromogen)
                 channel after colour deconvolution. DAB is the stain that
                 visualises HER2 protein, so the box does enclose genuinely
                 HER2-stained tissue -- but its exact outline is an algorithm's,
                 not a pathologist's.
    HER2 0     : no chromogen exists to localise, so the box is the whole tissue
                 area and the caption states the absence of staining.

Every emitted sample is logged to her2_meta.csv with the DAB-positive fraction
and whether a threshold fallback was needed, so a reviewer can audit how much
staining each heuristic box was actually drawn around.

Usage
    python preprocess_bci.py --bci_dir ../data/BCI_dataset/ --output_dir ../data/her2_512p/
'''

import argparse
import os
import re
import sys
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from her2 import boxes as her2_boxes
from her2 import captions as her2_captions
from her2 import manifest as her2_manifest
from her2 import stain as her2_stain
from tools.preprocess_utils import resize_with_padding, save_jpg

SEED = 0
FILENAME_PATTERN = re.compile(r'^(?P<id>\d+)_(?P<split>train|test)_(?P<grade>0|1\+|2\+|3\+)$')

# Thresholds tried in order when the primary one finds no region. A faint 1+
# membrane can sit below the default cut-off; rather than dropping the sample or
# silently inventing a box, we step down and record that we did.
FALLBACK_THRESHOLDS = [0.08, 0.05]


def parse_filename(filename):
    match = FILENAME_PATTERN.match(os.path.splitext(os.path.basename(filename))[0])
    if match is None:
        return None
    return match.group('id'), match.group('split'), match.group('grade')


def build_boxes(rgb, grade, args):
    '''
    return: (normalized boxes, dab_fraction, threshold_used)
    '''
    dab_fraction = her2_stain.dab_fraction(rgb, threshold=args.dab_threshold)

    if her2_captions.is_unstained(grade):
        return None, dab_fraction, None  # handled by the caller as a tissue box

    threshold = args.strong_dab_threshold if grade == '3+' else args.dab_threshold
    for candidate in [threshold] + FALLBACK_THRESHOLDS:
        if candidate > threshold:
            continue
        mask = her2_stain.dab_mask(rgb, threshold=candidate)
        mask = her2_boxes.clean_mask(mask, close_px=args.close_px, open_px=args.open_px)
        found = her2_boxes.mask_to_boxes(
            mask,
            area_threshold=args.area_threshold,
            min_side=args.min_box_side,
            max_boxes=args.max_boxes,
        )
        if len(found) > 0:
            return found, dab_fraction, candidate
    return [], dab_fraction, None


def preprocess_image(image_path, split, save_dir, args):
    '''
    return: (rel_path, boxes, label, meta_row) or None
    '''
    parsed = parse_filename(image_path)
    if parsed is None:
        return None
    patch_id, source_split, grade = parsed

    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    rgb = np.asarray(image)

    normalized, dab_frac, threshold_used = build_boxes(rgb, grade, args)

    if normalized is None:  # HER2 0: no chromogen to localise
        mask = her2_stain.tissue_mask(rgb, threshold=args.tissue_od_threshold)
        padded = her2_boxes.whole_tissue_box(mask, (width, height), args.target_size)
        box_kind = 'whole-tissue'
    elif len(normalized) == 0:
        # Graded as stained but nothing survived even the lowest threshold. This
        # is a genuine disagreement between label and pixels; drop it rather than
        # attach the grade text to an arbitrary box.
        return None
    else:
        padded = her2_boxes.to_padded_pixels(normalized, (width, height), args.target_size)
        box_kind = f'dab@{threshold_used:g}'

    if len(padded) == 0:
        return None

    label = her2_captions.her2_region_text(grade)
    filename = f'her2_bci_{source_split}_{patch_id}_{grade.replace("+", "p")}.jpg'
    save_jpg(resize_with_padding(image, args.target_size), os.path.join(save_dir, filename))

    rel_path = f'{split}/{filename}'
    meta_row = {
        'image': rel_path,
        'label': label,
        'source': f'BCI ({box_kind})',
        'content_type': her2_captions.CONTENT_TYPE_IHC,
        'grade': grade,
        'slide_id': f'bci_{source_split}_{patch_id}',
        'num_boxes': len(padded),
        'dab_fraction': round(dab_frac, 5),
    }
    return rel_path, padded, label, meta_row


def collect_images(bci_dir):
    '''
    return: {'train': [paths], 'test': [paths]} for the IHC arm of BCI
    '''
    found = {'train': [], 'test': []}
    ihc_root = os.path.join(bci_dir, 'IHC')
    if not os.path.isdir(ihc_root):
        raise ValueError(f'expected an IHC/ directory under {bci_dir}')
    for source_split in ('train', 'test'):
        split_dir = os.path.join(ihc_root, source_split)
        if not os.path.isdir(split_dir):
            continue
        for filename in sorted(os.listdir(split_dir)):
            if os.path.splitext(filename)[1].lower() not in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
                continue
            if parse_filename(filename) is None:
                continue
            found[source_split].append(os.path.join(split_dir, filename))
    return found


def target_split(source_split, patch_id, args):
    '''
    BCI's own test split is honoured as our test split. Validation is carved out
    of BCI train deterministically.

    Caveat, documented in README_her2.md: BCI publishes patch ids, not slide ids,
    so we cannot guarantee patch-level splits are slide-disjoint within the BCI
    train pool. The held-out test set is BCI's own and is unaffected.
    '''
    if source_split == 'test':
        return 'test'
    assigned = her2_manifest.assign_split(patch_id, seed=args.seed, val_ratio=args.val_ratio, test_ratio=0.0)
    return 'val' if assigned == 'val' else 'train'


def warn_token_budget(max_boxes):
    default_fit = her2_boxes.max_boxes_for(decoder_max_length=150, multimodal_tokens=128)
    if max_boxes > default_fit:
        needed = her2_boxes.required_decoder_max_length(max_boxes, multimodal_tokens=128)
        print(
            f'NOTE: --max_boxes {max_boxes} exceeds the {default_fit} boxes that fit the default\n'
            f'      --decoder_max_length 150. Train with --decoder_max_length {needed} or the\n'
            f'      end of the target sequence will be truncated. See README_her2.md, "Token budget".'
        )


def run(args):
    warn_token_budget(args.max_boxes)
    images = collect_images(args.bci_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    for split in her2_manifest.SPLITS:
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)

    tasks = []
    per_split = {split: 0 for split in her2_manifest.SPLITS}
    for source_split, paths in images.items():
        for image_path in paths:
            patch_id, _, _ = parse_filename(image_path)
            split = target_split(source_split, patch_id, args)
            if args.limit_per_split and per_split[split] >= args.limit_per_split:
                continue
            per_split[split] += 1
            tasks.append((image_path, split, os.path.join(args.output_dir, split), args))

    print(f'BCI IHC patches to convert: {len(tasks)} {per_split}')

    if args.num_workers <= 1 or len(tasks) <= 1:
        results = [preprocess_image(*task) for task in tqdm(tasks, desc='preprocess bci')]
    else:
        with Pool(processes=min(args.num_workers, len(tasks))) as pool:
            results = pool.starmap(preprocess_image, tasks)

    manifests = {split: her2_manifest.new_manifest(args.modality) for split in her2_manifest.SPLITS}
    meta_rows = []
    dropped = 0
    for result in results:
        if result is None:
            dropped += 1
            continue
        rel_path, padded, label, meta_row = result
        split = rel_path.split('/')[0]
        her2_manifest.add_sample(manifests[split], rel_path, padded, label)
        meta_rows.append(meta_row)

    for split in her2_manifest.SPLITS:
        if len(manifests[split]['data']) == 0:
            continue
        her2_manifest.validate_manifest(manifests[split], split, args.output_dir, args.target_size)
        path = her2_manifest.write_manifest(manifests[split], args.output_dir, f'{split}_bci')
        print(f'{split}: {len(manifests[split]["data"])} images -> {path}')

    her2_manifest.write_meta(meta_rows, args.output_dir)
    print(f'Converted: {len(meta_rows)} | Dropped (no localisable staining): {dropped}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bci_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--target_size', type=int)
    parser.add_argument('--modality', type=str)
    parser.add_argument('--dab_threshold', type=float)
    parser.add_argument('--strong_dab_threshold', type=float)
    parser.add_argument('--tissue_od_threshold', type=float)
    parser.add_argument('--area_threshold', type=float)
    parser.add_argument('--min_box_side', type=float)
    parser.add_argument('--max_boxes', type=int)
    parser.add_argument('--close_px', type=int)
    parser.add_argument('--open_px', type=int)
    parser.add_argument('--val_ratio', type=float)
    parser.add_argument('--limit_per_split', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.set_defaults(
        bci_dir='../data/BCI_dataset/',
        output_dir='../data/her2_512p/',
        target_size=512,
        modality='Histopathology (IHC)',
        dab_threshold=her2_stain.DAB_TISSUE_THRESHOLD,
        strong_dab_threshold=her2_stain.DAB_STRONG_THRESHOLD,
        tissue_od_threshold=0.10,
        area_threshold=0.06,
        min_box_side=her2_boxes.MIN_BOX_SIDE,
        max_boxes=her2_boxes.MAX_BOXES,
        close_px=5,
        open_px=3,
        val_ratio=0.1,
        limit_per_split=0,
        seed=SEED,
        num_workers=max(1, cpu_count() // 2),
    )
    args = parser.parse_args()
    run(args)
