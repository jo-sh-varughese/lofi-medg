'''
Build the lesion-polygon track of the `her2` dataset from CAMELYON16/17.

This is the only track whose boxes are real. CAMELYON ships ASAP XML polygons
drawn by pathologists around metastatic regions, which convert directly into
grounding boxes the same way SegTHOR masks do.

Supervision honesty
-------------------
    box        : REAL. Derived from pathologist-drawn polygons.
    label text : REAL but generic. It states metastatic carcinoma versus benign
                 lymph node tissue. It carries NO HER2 claim, because CAMELYON
                 is H&E and contains no HER2 information whatsoever.

Purpose of this track: it teaches the model histopathology-shaped region
grounding on trustworthy geometry, which the templated IHC track cannot. Both
tracks land in one `her2_512p/` directory and are distinguished by content_type
so evaluation can be reported separately.

Usage
    python preprocess_camelyon.py \
        --slide_dir ../data/CAMELYON16/images/ \
        --annotation_dir ../data/CAMELYON16/annotations/ \
        --output_dir ../data/her2_512p/
'''

import argparse
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from her2 import boxes as her2_boxes
from her2 import captions as her2_captions
from her2 import manifest as her2_manifest
from her2 import wsi as her2_wsi
from tools import preprocess_bci
from tools.preprocess_utils import resize_with_padding, save_jpg

SEED = 0
SLIDE_EXTENSIONS = ('.tif', '.tiff', '.svs', '.ndpi', '.mrxs')


def slide_id_of(slide_path):
    '''
    CAMELYON17 groups slides by patient (`patient_004_node_4.tif`); splitting on
    the patient keeps all of a patient's nodes in one split.
    '''
    stem = os.path.splitext(os.path.basename(slide_path))[0]
    parts = stem.split('_')
    if len(parts) >= 2 and parts[0].lower() == 'patient':
        return f'patient_{parts[1]}'
    return stem


def find_slides(slide_dir):
    slides = []
    for root, _, filenames in os.walk(slide_dir):
        for filename in sorted(filenames):
            if os.path.splitext(filename)[1].lower() in SLIDE_EXTENSIONS:
                slides.append(os.path.join(root, filename))
    return sorted(slides)


def preprocess_slide(slide_path, annotation_dir, split, save_dir, args):
    '''
    return: (list of (rel_path, boxes, label, meta_row))
    '''
    slide = her2_wsi.open_slide(slide_path)
    annotation_path = her2_wsi.find_annotation_file(annotation_dir, slide_path)
    annotations = her2_wsi.read_asap_annotations(annotation_path) if annotation_path else []
    annotations = [(group, points) for group, points in annotations if her2_captions.lesion_region_text(group)]

    stem = os.path.splitext(os.path.basename(slide_path))[0]
    outputs = []
    tile_limit = args.tiles_per_slide if annotations else args.negative_tiles_per_slide

    for base_x, base_y, _, downsample, tile in her2_wsi.iter_tiles(
        slide,
        tile_size=args.tile_size,
        target_mpp=args.target_mpp,
        stride=args.stride or args.tile_size,
        tissue_ratio_min=args.tissue_ratio,
        limit=None,
    ):
        window = args.tile_size * downsample
        clipped = her2_wsi.polygons_in_window(annotations, base_x, base_y, window, window, downsample=downsample)

        if clipped:
            # 'tumor' and 'metastases' are the same lesion class under different
            # CAMELYON16/17 group names, so they are pooled by caption text
            groups = {group for group, _ in clipped}
            group = 'metastases' if groups & {'metastases', 'tumor'} else sorted(groups)[0]
            wanted = her2_captions.lesion_region_text(group)
            normalized = her2_boxes.polygons_to_boxes(
                [points for name, points in clipped if her2_captions.lesion_region_text(name) == wanted],
                args.tile_size, args.tile_size,
                min_side=args.min_box_side, max_boxes=args.max_boxes,
            )
        elif annotations:
            continue  # annotated slide: skip tiles with no lesion rather than guess
        else:
            group = 'normal'
            normalized = [[0.0, 0.0, 1.0, 1.0]]

        if len(normalized) == 0:
            continue

        padded = her2_boxes.to_padded_pixels(normalized, (args.tile_size, args.tile_size), args.target_size)
        if len(padded) == 0:
            continue

        label = her2_captions.lesion_region_text(group)
        filename = f'her2_camelyon_{stem}_{base_x}_{base_y}.jpg'
        save_jpg(resize_with_padding(tile, args.target_size), os.path.join(save_dir, filename))

        rel_path = f'{split}/{filename}'
        outputs.append((rel_path, padded, label, {
            'image': rel_path,
            'label': label,
            'source': 'CAMELYON' if annotations else 'CAMELYON (negative tile)',
            'content_type': her2_captions.CONTENT_TYPE_LESION,
            'grade': '',
            'slide_id': slide_id_of(slide_path),
            'num_boxes': len(padded),
            'dab_fraction': '',
        }))

        if tile_limit and len(outputs) >= tile_limit:
            break

    slide.close()
    return outputs


def run(args):
    preprocess_bci.warn_token_budget(args.max_boxes)
    slides = find_slides(args.slide_dir)
    if len(slides) == 0:
        raise ValueError(f'no whole-slide images found under {args.slide_dir}')
    print(f'CAMELYON slides: {len(slides)}')

    os.makedirs(args.output_dir, exist_ok=True)
    for split in her2_manifest.SPLITS:
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)

    manifests = {split: her2_manifest.new_manifest(args.modality) for split in her2_manifest.SPLITS}
    meta_rows = []

    for slide_path in tqdm(slides, desc='preprocess camelyon'):
        split = her2_manifest.assign_split(
            slide_id_of(slide_path), seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        )
        save_dir = os.path.join(args.output_dir, split)
        try:
            outputs = preprocess_slide(slide_path, args.annotation_dir, split, save_dir, args)
        except Exception as error:  # a single unreadable slide must not sink the run
            print(f'Skipping {os.path.basename(slide_path)}: {error}')
            continue
        for rel_path, padded, label, meta_row in outputs:
            her2_manifest.add_sample(manifests[split], rel_path, padded, label)
            meta_rows.append(meta_row)

    for split in her2_manifest.SPLITS:
        if len(manifests[split]['data']) == 0:
            continue
        her2_manifest.validate_manifest(manifests[split], split, args.output_dir, args.target_size)
        path = her2_manifest.write_manifest(manifests[split], args.output_dir, f'{split}_camelyon')
        print(f'{split}: {len(manifests[split]["data"])} images -> {path}')

    her2_manifest.write_meta(meta_rows, args.output_dir)
    print(f'Converted tiles: {len(meta_rows)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_dir', type=str)
    parser.add_argument('--annotation_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--target_size', type=int)
    parser.add_argument('--modality', type=str)
    parser.add_argument('--tile_size', type=int)
    parser.add_argument('--stride', type=int)
    parser.add_argument('--target_mpp', type=float)
    parser.add_argument('--tissue_ratio', type=float)
    parser.add_argument('--tiles_per_slide', type=int)
    parser.add_argument('--negative_tiles_per_slide', type=int)
    parser.add_argument('--min_box_side', type=float)
    parser.add_argument('--max_boxes', type=int)
    parser.add_argument('--val_ratio', type=float)
    parser.add_argument('--test_ratio', type=float)
    parser.add_argument('--seed', type=int)
    parser.set_defaults(
        slide_dir='../data/CAMELYON16/images/',
        annotation_dir='../data/CAMELYON16/annotations/',
        output_dir='../data/her2_512p/',
        target_size=512,
        modality='Histopathology (H&E)',
        tile_size=1024,
        stride=1024,
        target_mpp=her2_wsi.DEFAULT_MPP,
        tissue_ratio=her2_wsi.DEFAULT_TISSUE_RATIO,
        tiles_per_slide=40,
        negative_tiles_per_slide=10,
        min_box_side=her2_boxes.MIN_BOX_SIDE,
        max_boxes=her2_boxes.MAX_BOXES,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=SEED,
    )
    args = parser.parse_args()
    run(args)
