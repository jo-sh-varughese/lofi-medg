import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

# Colours of a synthetic IHC field. These are not real tissue -- they exist only
# so the format, coordinate and threshold logic can be tested without shipping or
# downloading patient data.
GLASS = (242, 242, 240)
HEMATOXYLIN_BLUE = (120, 110, 180)
DAB_BROWN = (135, 85, 40)
DAB_FAINT = (190, 165, 140)


def _disc(array, center_x, center_y, radius, color):
    height, width = array.shape[:2]
    ys, xs = np.ogrid[:height, :width]
    mask = (xs - center_x) ** 2 + (ys - center_y) ** 2 <= radius ** 2
    array[mask] = color


def make_ihc_patch(grade, size=256, seed=0):
    '''
    Synthetic HER2 IHC field: blue nuclei on glass, plus brown membrane blobs
    whose darkness follows the requested grade.
    '''
    rng = np.random.default_rng(seed)
    array = np.full((size, size, 3), GLASS, dtype=np.uint8)

    for _ in range(24):
        _disc(array, rng.integers(10, size - 10), rng.integers(10, size - 10), 6, HEMATOXYLIN_BLUE)

    if grade != '0':
        color = DAB_BROWN if grade == '3+' else DAB_FAINT if grade == '1+' else tuple(
            (np.array(DAB_BROWN) + np.array(DAB_FAINT)) // 2
        )
        radius = {'1+': 26, '2+': 32, '3+': 40}[grade]
        for center in [(size // 4, size // 4), (3 * size // 4, 2 * size // 3)]:
            _disc(array, center[0], center[1], radius, color)

    return array


@pytest.fixture
def bci_dir(tmp_path):
    '''
    A miniature BCI-shaped directory: IHC/{train,test}/<id>_<split>_<grade>.png
    '''
    root = tmp_path / 'BCI_dataset'
    grades = ['0', '1+', '2+', '3+']
    for split in ('train', 'test'):
        (root / 'IHC' / split).mkdir(parents=True)
        for index, grade in enumerate(grades):
            patch = make_ihc_patch(grade, seed=index + (0 if split == 'train' else 100))
            filename = f'{index:05d}_{split}_{grade}.png'
            Image.fromarray(patch).save(root / 'IHC' / split / filename)
    return str(root)


@pytest.fixture
def det_source():
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'lofi_utils', 'dataset', 'det.py')
    with open(path, 'rt', encoding='utf-8') as f:
        return f.read()
