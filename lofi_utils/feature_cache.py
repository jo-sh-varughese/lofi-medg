'''
Vision-feature caching for frozen-encoder fine-tuning.

WHY THIS IS EXACT, NOT AN APPROXIMATION
    The downstream recipe passes --fix_enc, which sets requires_grad=False on
    every encoder parameter (main.py:185-187). There is also no image
    augmentation anywhere in the pipeline: BaseDataset.__getitem__ loads the
    image and hands it straight to the processor, with no random crop, flip or
    jitter. So for a fixed checkpoint the encoder output for a given image is
    the same tensor on every epoch, and computing it 30 times is pure waste.

    This module computes it once and stores it. The text target still varies
    per epoch (base.py:62 picks the grounding/captioning direction at random),
    and the projection head is still trained -- the cache sits strictly
    upstream of both, so neither is affected.

WHAT IS CACHED
    model.vision_model(pixel_values).last_hidden_state, i.e. exactly what
    train_eval.encode_vision returns. With --pool2x2 the parameter-free 2x2
    average pool (model.pool2x2) is applied before storing, which cuts the
    token count 1024 -> 256 and the cache size by 4x. ProjectionWrapper is then
    told to skip its own pooling for cached input.

VALIDITY
    A cache is only valid for the exact (encoder checkpoint, resumed weights,
    image size, pooling) combination that produced it. build_manifest records
    those and check_manifest refuses a mismatch loudly rather than silently
    training on stale features. Deleting the cache directory is always safe.
'''

import hashlib
import json
import os

import numpy as np

MANIFEST_NAME = 'manifest.json'
SUPPORTED_DTYPES = ('float16', 'float32')


def cache_key(image_path):
    '''
    Stable key for an image. Uses the absolute, normalised, case-folded path so
    the same file does not get two entries via a relative vs. absolute path, or
    via Windows path-case differences.
    '''
    normalised = os.path.normcase(os.path.abspath(image_path)).replace('\\', '/')
    return hashlib.sha1(normalised.encode('utf-8')).hexdigest()


def cache_file_path(cache_dir, image_path):
    '''
    Shard on the first two hex characters. A flat directory with tens of
    thousands of files is slow to list on Windows and on network drives.
    '''
    key = cache_key(image_path)
    return os.path.join(cache_dir, key[:2], f'{key}.npy')


def fingerprint_file(path):
    '''
    Cheap identity for a checkpoint: size plus mtime. Hashing a multi-GB
    last.pt on every run would cost more than it is worth, and this is a
    staleness guard, not a security boundary.
    '''
    if not path or not os.path.isfile(path):
        return None
    stat = os.stat(path)
    return {'path': os.path.basename(path), 'size': stat.st_size, 'mtime': int(stat.st_mtime)}


def build_manifest(model_name, resume_path, image_size, pool2x2, dtype, feature_shape):
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f'unsupported cache dtype {dtype!r}, expected one of {SUPPORTED_DTYPES}')
    return {
        'model_name': model_name,
        'resume': fingerprint_file(resume_path),
        'image_size': int(image_size),
        'pool2x2': bool(pool2x2),
        'dtype': dtype,
        'feature_shape': [int(v) for v in feature_shape],
    }


def check_manifest(expected, found):
    '''
    Return a list of human-readable mismatches. Empty list means the cache is
    usable. Kept as a pure function so it is testable without torch.
    '''
    problems = []
    for key in ('model_name', 'image_size', 'pool2x2', 'dtype'):
        if expected.get(key) != found.get(key):
            problems.append(f'{key}: cache was built with {found.get(key)!r}, this run wants {expected.get(key)!r}')

    expected_resume, found_resume = expected.get('resume'), found.get('resume')
    if expected_resume != found_resume:
        problems.append(
            f'resume checkpoint differs: cache was built from {found_resume!r}, this run uses {expected_resume!r}. '
            f'Features from a different checkpoint are not interchangeable.'
        )
    return problems


def read_manifest(cache_dir):
    path = os.path.join(cache_dir, MANIFEST_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def write_manifest(cache_dir, manifest):
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, MANIFEST_NAME), 'wt', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


class FeatureCache:
    '''
    Read-only view over a built cache, handed to the dataset.

    Deliberately fails loudly on a missing entry. A silent fallback to running
    the encoder would make a half-built cache look like it worked while
    quietly costing full price, which is the bug this whole module exists to
    avoid.
    '''

    def __init__(self, cache_dir, expected_manifest=None):
        self.cache_dir = cache_dir
        self.manifest = read_manifest(cache_dir)
        if self.manifest is None:
            raise FileNotFoundError(
                f'no {MANIFEST_NAME} in {cache_dir}; build the cache first with tools/precompute_features.py'
            )
        if expected_manifest is not None:
            problems = check_manifest(expected_manifest, self.manifest)
            if problems:
                raise ValueError(
                    'feature cache does not match this run:\n  - ' + '\n  - '.join(problems) +
                    f'\nRebuild it, or point --feature_cache_dir somewhere else.'
                )

    @property
    def pooled(self):
        return bool(self.manifest.get('pool2x2'))

    def has(self, image_path):
        return os.path.isfile(cache_file_path(self.cache_dir, image_path))

    def load(self, image_path):
        path = cache_file_path(self.cache_dir, image_path)
        if not os.path.isfile(path):
            raise KeyError(
                f'{image_path} is not in the feature cache at {self.cache_dir}. '
                f'The cache is incomplete -- rerun tools/precompute_features.py.'
            )
        return np.load(path)

    def missing(self, image_paths):
        '''Unique image paths with no cache entry, in first-seen order.'''
        seen, out = set(), []
        for path in image_paths:
            if path in seen:
                continue
            seen.add(path)
            if not self.has(path):
                out.append(path)
        return out


def save_feature(cache_dir, image_path, array):
    path = cache_file_path(cache_dir, image_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Write to a temporary name then replace, so an interrupted run leaves no
    # half-written .npy that would load as corrupt on the next pass.
    # (the .npy suffix is kept so np.save does not append one of its own)
    tmp_path = path + '.tmp.npy'
    np.save(tmp_path, array, allow_pickle=False)
    os.replace(tmp_path, path)


def unique_images(dataset):
    '''
    Dataset entries repeat an image once per question/answer pair, so encoding
    per sample would redo identical work. Deduplicate first.
    '''
    seen, out = set(), []
    for path in dataset.imgs:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def estimate_cache_bytes(n_images, feature_shape, dtype):
    itemsize = 2 if dtype == 'float16' else 4
    tokens, width = feature_shape[-2], feature_shape[-1]
    return n_images * tokens * width * itemsize
