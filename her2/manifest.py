'''
Manifest construction and validation for the HER2 dataset.

The manifest is byte-for-byte the schema `lofi_utils/dataset/det.py` already
reads for `tn5000` and `segthor`:

    {
      "modality": "Histopathology",
      "data": {
        "train/her2_bci_00001.jpg": [
          {"box": [[x0, y0, x1, y1], ...], "label": "strong complete membranous staining, HER2 3+ positive"}
        ]
      }
    }

Coordinates are integer pixels on the letterboxed 512x512 canvas. Each entry in
the list under an image is one independent training sample.

`validate_manifest` re-implements the loader's arithmetic and fails loudly on any
deviation, so a format regression is caught by the test suite rather than by a
silently degenerate training run.
'''

import hashlib
import json
import os

SPLITS = ('train', 'val', 'test')

# Sidecar columns. This is not read by the LoFi loader; it exists so every
# generated sample can be traced back to its source and its supervision strength.
META_FIELDS = ['image', 'label', 'source', 'content_type', 'grade', 'slide_id', 'num_boxes', 'dab_fraction']


def new_manifest(modality='Histopathology'):
    return {'modality': modality, 'data': {}}


def add_sample(manifest, rel_path, boxes, label):
    '''
    manifest: dict from new_manifest
    rel_path: path relative to the dataset dir, including the split folder
    boxes: [[int, int, int, int], ...] on the padded canvas, sorted by x0
    label: region description text
    '''
    if len(boxes) == 0:
        raise ValueError(f'refusing to add a sample with no boxes: {rel_path}')
    if not str(label).strip():
        raise ValueError(f'refusing to add a sample with empty label: {rel_path}')
    manifest['data'].setdefault(rel_path, []).append({'box': boxes, 'label': label})


def write_manifest(manifest, output_dir, split):
    path = os.path.join(output_dir, f'{split}.json')
    with open(path, 'wt', encoding='utf-8') as wf:
        json.dump(manifest, wf, indent=2)
    return path


def read_manifest(output_dir, split):
    with open(os.path.join(output_dir, f'{split}.json'), 'rt', encoding='utf-8') as f:
        return json.load(f)


def write_meta(rows, output_dir, filename='her2_meta.csv'):
    '''
    rows: list of dicts with keys from META_FIELDS
    '''
    import pandas as pd

    path = os.path.join(output_dir, filename)
    frame = pd.DataFrame(rows, columns=META_FIELDS)
    if os.path.isfile(path):
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    frame.to_csv(path, index=False)
    return path


def assign_split(key, seed=0, val_ratio=0.1, test_ratio=0.1):
    '''
    Deterministic split assignment keyed on a *slide or case* identifier, never on
    a patch identifier, so patches from one slide cannot straddle two splits.

    key: slide/case id
    return: 'train' | 'val' | 'test'
    '''
    digest = hashlib.sha256(f'{seed}:{key}'.encode('utf-8')).hexdigest()
    position = int(digest[:8], 16) / 0xFFFFFFFF
    if position < test_ratio:
        return 'test'
    if position < test_ratio + val_ratio:
        return 'val'
    return 'train'


def merge_manifests(manifests, modality='Histopathology'):
    '''
    Combine per-track manifests for one split into a single manifest.

    manifests: list of manifest dicts
    '''
    merged = new_manifest(modality)
    for manifest in manifests:
        for rel_path, pairs in manifest.get('data', {}).items():
            if rel_path in merged['data']:
                raise ValueError(f'duplicate image path across tracks: {rel_path}')
            merged['data'][rel_path] = pairs
    return merged


def validate_manifest(manifest, split, dataset_dir=None, target_size=512, check_images=True):
    '''
    Assert the manifest is exactly what `det.py` expects. Raises ValueError with a
    specific message on the first violation.

    Mirrors the loader:
        boxes = sorted(pair['box'], key=lambda box: box[0])
        boxes = str([[int((v / target_annotated_size) * 1000) for v in box] for box in boxes]).replace(' ', '')

    return: number of samples validated
    '''
    if not isinstance(manifest, dict):
        raise ValueError('manifest must be a dict')
    if 'modality' not in manifest:
        raise ValueError('manifest is missing "modality"')
    if not isinstance(manifest.get('data'), dict):
        raise ValueError('manifest is missing a "data" object')

    samples = 0
    for rel_path, pairs in manifest['data'].items():
        if os.path.isabs(rel_path) or '\\' in rel_path:
            raise ValueError(f'image key must be a relative POSIX path: {rel_path}')
        if not rel_path.startswith(f'{split}/'):
            raise ValueError(f'image key must start with "{split}/": {rel_path}')
        if not rel_path.endswith('.jpg'):
            raise ValueError(f'image key must be a .jpg: {rel_path}')
        if check_images and dataset_dir is not None:
            if not os.path.isfile(os.path.join(dataset_dir, rel_path)):
                raise ValueError(f'image referenced by the manifest is missing: {rel_path}')

        if not isinstance(pairs, list) or len(pairs) == 0:
            raise ValueError(f'image must map to a non-empty list of pairs: {rel_path}')

        for pair in pairs:
            if set(pair.keys()) != {'box', 'label'}:
                raise ValueError(f'pair must have exactly "box" and "label" keys: {rel_path} -> {sorted(pair.keys())}')
            if not isinstance(pair['label'], str) or not pair['label'].strip():
                raise ValueError(f'label must be a non-empty string: {rel_path}')
            if pair['label'].endswith('.'):
                # The loader strips one trailing period when building the prompt;
                # keeping labels period-free means the text is identical in the
                # grounding and captioning directions.
                raise ValueError(f'label must not end with a period: {rel_path} -> {pair["label"]!r}')

            boxes = pair['box']
            if not isinstance(boxes, list) or len(boxes) == 0:
                raise ValueError(f'box must be a non-empty list of boxes: {rel_path}')
            previous_x0 = -1
            for box in boxes:
                if not isinstance(box, list) or len(box) != 4:
                    raise ValueError(f'box must have 4 values: {rel_path} -> {box}')
                if not all(isinstance(v, int) for v in box):
                    raise ValueError(f'box values must be Python ints, not floats: {rel_path} -> {box}')
                x0, y0, x1, y1 = box
                if x1 <= x0 or y1 <= y0:
                    raise ValueError(f'box must satisfy x1 > x0 and y1 > y0: {rel_path} -> {box}')
                if min(box) < 0 or max(box) > target_size:
                    raise ValueError(f'box must lie within [0, {target_size}]: {rel_path} -> {box}')
                if x0 < previous_x0:
                    raise ValueError(f'boxes must be sorted by x0: {rel_path} -> {boxes}')
                previous_x0 = x0

                # the loader's own arithmetic must stay inside the 0-1000 bins
                binned = [int((v / target_size) * 1000) for v in box]
                if min(binned) < 0 or max(binned) > 1000:
                    raise ValueError(f'box falls outside the 0-1000 decoder bins: {rel_path} -> {binned}')

            samples += 1

    if samples == 0:
        raise ValueError(f'manifest for split "{split}" contains no samples')
    return samples
