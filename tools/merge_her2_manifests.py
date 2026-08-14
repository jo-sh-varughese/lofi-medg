'''
Merge the per-track HER2 manifests into the split files the loader reads.

The two preprocessing scripts write `<split>_bci.json` and `<split>_camelyon.json`
so each track can be rebuilt independently. This step combines whichever tracks
exist into `<split>.json`, then re-validates the result against the loader's
expectations.

Usage
    python merge_her2_manifests.py --output_dir ../data/her2_512p/
'''

import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from her2 import manifest as her2_manifest

TRACKS = ('bci', 'camelyon')


def run(args):
    summary = {}
    for split in her2_manifest.SPLITS:
        parts = []
        for track in TRACKS:
            path = os.path.join(args.output_dir, f'{split}_{track}.json')
            if not os.path.isfile(path):
                continue
            with open(path, 'rt', encoding='utf-8') as f:
                parts.append(json.load(f))

        if len(parts) == 0:
            print(f'{split}: no track manifests found, skipping')
            continue

        merged = her2_manifest.merge_manifests(parts, modality=args.modality)
        samples = her2_manifest.validate_manifest(
            merged, split, args.output_dir, args.target_size, check_images=not args.no_check_images,
        )
        path = her2_manifest.write_manifest(merged, args.output_dir, split)
        summary[split] = {'images': len(merged['data']), 'samples': samples}
        print(f'{split}: {len(merged["data"])} images / {samples} samples -> {path}')

    if len(summary) == 0:
        raise ValueError(f'nothing to merge under {args.output_dir}; run a preprocess script first')

    with open(os.path.join(args.output_dir, 'her2_summary.json'), 'wt', encoding='utf-8') as wf:
        json.dump(summary, wf, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--target_size', type=int)
    parser.add_argument('--modality', type=str)
    parser.add_argument('--no_check_images', action='store_true')
    parser.set_defaults(
        output_dir='../data/her2_512p/',
        target_size=512,
        modality='Histopathology',
    )
    args = parser.parse_args()
    run(args)
