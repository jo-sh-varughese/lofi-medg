import copy

import pytest

from her2 import manifest as her2_manifest

GOOD = {
    'modality': 'Histopathology',
    'data': {
        'train/her2_bci_train_00001_3p.jpg': [
            {'box': [[10, 20, 100, 120], [200, 30, 300, 140]], 'label': 'strong complete membranous staining, HER2 3+ positive'}
        ]
    },
}


def validate(manifest, split='train'):
    return her2_manifest.validate_manifest(manifest, split, dataset_dir=None, check_images=False)


def test_accepts_a_well_formed_manifest():
    assert validate(copy.deepcopy(GOOD)) == 1


def test_rejects_float_coordinates():
    bad = copy.deepcopy(GOOD)
    bad['data']['train/her2_bci_train_00001_3p.jpg'][0]['box'] = [[10.0, 20.0, 100.0, 120.0]]
    with pytest.raises(ValueError, match='Python ints'):
        validate(bad)


def test_rejects_unsorted_boxes():
    bad = copy.deepcopy(GOOD)
    bad['data']['train/her2_bci_train_00001_3p.jpg'][0]['box'] = [[200, 30, 300, 140], [10, 20, 100, 120]]
    with pytest.raises(ValueError, match='sorted by x0'):
        validate(bad)


def test_rejects_degenerate_boxes():
    bad = copy.deepcopy(GOOD)
    bad['data']['train/her2_bci_train_00001_3p.jpg'][0]['box'] = [[100, 20, 100, 120]]
    with pytest.raises(ValueError, match='x1 > x0'):
        validate(bad)


def test_rejects_boxes_outside_the_canvas():
    bad = copy.deepcopy(GOOD)
    bad['data']['train/her2_bci_train_00001_3p.jpg'][0]['box'] = [[10, 20, 513, 120]]
    with pytest.raises(ValueError, match=r'within \[0, 512\]'):
        validate(bad)


def test_rejects_wrong_split_prefix():
    with pytest.raises(ValueError, match='must start with "val/"'):
        validate(copy.deepcopy(GOOD), split='val')


def test_rejects_trailing_period_in_label():
    bad = copy.deepcopy(GOOD)
    bad['data']['train/her2_bci_train_00001_3p.jpg'][0]['label'] = 'HER2 3+ positive.'
    with pytest.raises(ValueError, match='must not end with a period'):
        validate(bad)


def test_rejects_extra_keys_in_a_pair():
    bad = copy.deepcopy(GOOD)
    bad['data']['train/her2_bci_train_00001_3p.jpg'][0]['grade'] = '3+'
    with pytest.raises(ValueError, match='exactly "box" and "label"'):
        validate(bad)


def test_rejects_empty_manifest():
    with pytest.raises(ValueError, match='no samples'):
        validate(her2_manifest.new_manifest())


def test_add_sample_refuses_empty_boxes_or_labels():
    manifest = her2_manifest.new_manifest()
    with pytest.raises(ValueError):
        her2_manifest.add_sample(manifest, 'train/a.jpg', [], 'text')
    with pytest.raises(ValueError):
        her2_manifest.add_sample(manifest, 'train/a.jpg', [[1, 2, 3, 4]], '  ')


def test_merge_rejects_duplicate_images():
    with pytest.raises(ValueError, match='duplicate image path'):
        her2_manifest.merge_manifests([copy.deepcopy(GOOD), copy.deepcopy(GOOD)])


def test_split_assignment_is_deterministic_and_slide_keyed():
    first = her2_manifest.assign_split('patient_004')
    assert first == her2_manifest.assign_split('patient_004')
    assert first in her2_manifest.SPLITS

    assignments = [her2_manifest.assign_split(f'slide_{i}') for i in range(2000)]
    test_ratio = assignments.count('test') / len(assignments)
    val_ratio = assignments.count('val') / len(assignments)
    assert 0.07 < test_ratio < 0.13
    assert 0.07 < val_ratio < 0.13
