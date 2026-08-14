'''
End-to-end conversion of a handful of synthetic IHC patches, plus the checks that
tie the output to what `lofi_utils/dataset/det.py` actually reads.
'''

import json
import os
import re
from types import SimpleNamespace

import pandas as pd
from PIL import Image

from her2 import boxes as her2_boxes
from her2 import captions as her2_captions
from her2 import manifest as her2_manifest
from her2 import stain as her2_stain
from tools import merge_her2_manifests, preprocess_bci


def build_args(bci_dir, output_dir):
    return SimpleNamespace(
        bci_dir=bci_dir,
        output_dir=output_dir,
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
        seed=0,
        num_workers=1,
    )


def convert(bci_dir, tmp_path):
    output_dir = str(tmp_path / 'her2_512p')
    preprocess_bci.run(build_args(bci_dir, output_dir))
    merge_her2_manifests.run(SimpleNamespace(
        output_dir=output_dir, target_size=512, modality='Histopathology', no_check_images=False,
    ))
    return output_dir


def test_conversion_produces_a_loader_ready_dataset(bci_dir, tmp_path):
    output_dir = convert(bci_dir, tmp_path)

    # the directory name must encode 512, because get_target_annotated_size()
    # parses the resolution out of it and silently defaults otherwise
    assert re.search(r'_(\d+)p$', os.path.basename(output_dir))

    total = 0
    for split in her2_manifest.SPLITS:
        path = os.path.join(output_dir, f'{split}.json')
        if not os.path.isfile(path):
            continue
        manifest = her2_manifest.read_manifest(output_dir, split)
        total += her2_manifest.validate_manifest(manifest, split, output_dir, 512)
        assert manifest['modality']

    assert total > 0


def test_every_image_is_a_512_square_jpeg(bci_dir, tmp_path):
    output_dir = convert(bci_dir, tmp_path)
    manifest = her2_manifest.read_manifest(output_dir, 'test')
    for rel_path in manifest['data']:
        with Image.open(os.path.join(output_dir, rel_path)) as image:
            assert image.size == (512, 512)
            assert image.format == 'JPEG'


def test_every_caption_comes_from_the_template_vocabulary(bci_dir, tmp_path):
    output_dir = convert(bci_dir, tmp_path)
    vocabulary = {text for _, text in her2_captions.all_templates()}
    for split in her2_manifest.SPLITS:
        if not os.path.isfile(os.path.join(output_dir, f'{split}.json')):
            continue
        manifest = her2_manifest.read_manifest(output_dir, split)
        for pairs in manifest['data'].values():
            for pair in pairs:
                assert pair['label'] in vocabulary


def test_grade_zero_gets_a_whole_tissue_box_and_negative_text(bci_dir, tmp_path):
    output_dir = convert(bci_dir, tmp_path)
    meta = pd.read_csv(os.path.join(output_dir, 'her2_meta.csv'))

    zero_rows = meta[meta['grade'].astype(str) == '0']
    assert len(zero_rows) > 0
    for _, row in zero_rows.iterrows():
        assert row['label'] == her2_captions.HER2_REGION_TEXT['0']
        assert row['num_boxes'] == 1
        assert 'whole-tissue' in row['source']


def test_stained_grades_get_chromogen_boxes(bci_dir, tmp_path):
    output_dir = convert(bci_dir, tmp_path)
    meta = pd.read_csv(os.path.join(output_dir, 'her2_meta.csv'))

    stained = meta[meta['grade'].astype(str) == '3+']
    assert len(stained) > 0
    for _, row in stained.iterrows():
        assert 'dab@' in row['source']
        assert row['dab_fraction'] > 0
        assert row['label'] == her2_captions.HER2_REGION_TEXT['3+']


def test_meta_covers_every_manifest_image(bci_dir, tmp_path):
    output_dir = convert(bci_dir, tmp_path)
    meta = pd.read_csv(os.path.join(output_dir, 'her2_meta.csv'))
    tracked = set(meta['image'])

    for split in her2_manifest.SPLITS:
        if not os.path.isfile(os.path.join(output_dir, f'{split}.json')):
            continue
        manifest = her2_manifest.read_manifest(output_dir, split)
        assert set(manifest['data']) <= tracked


def test_official_bci_test_patches_stay_in_test(bci_dir, tmp_path):
    output_dir = convert(bci_dir, tmp_path)
    meta = pd.read_csv(os.path.join(output_dir, 'her2_meta.csv'))
    for _, row in meta.iterrows():
        if '_test_' in row['slide_id']:
            assert row['image'].startswith('test/')
        else:
            assert not row['image'].startswith('test/')


def test_box_string_matches_what_the_decoder_is_trained_to_emit(bci_dir, tmp_path):
    '''
    Reproduces det.py's serialisation and asserts the exact target string.
    '''
    output_dir = convert(bci_dir, tmp_path)
    manifest = her2_manifest.read_manifest(output_dir, 'test')

    for pairs in manifest['data'].values():
        for pair in pairs:
            boxes = sorted(pair['box'], key=lambda box: box[0])
            encoded = str([[int((v / 512) * 1000) for v in box] for box in boxes]).replace(' ', '')

            assert ' ' not in encoded
            assert encoded.startswith('[[') and encoded.endswith(']]')
            for value in json.loads(encoded):
                assert all(0 <= v <= 1000 for v in value)


def test_her2_loader_encodes_boxes_identically_to_segthor(det_source):
    '''
    Guards against the HER2 loader drifting from the convention the released
    checkpoint was trained under.
    '''
    encode_line = "boxes = str([[int((v / target_annotated_size) * 1000) for v in box] for box in boxes]).replace(' ', '')"
    assert det_source.count(encode_line) >= 4  # medg, tn5000, segthor, her2

    her2_block = det_source.split('class HER2Dataset')[1].split('\nclass ')[0]
    assert encode_line in her2_block
    assert "sorted(pair['box'], key=lambda box: box[0])" in her2_block
    assert 'get_target_annotated_size(args.her2_dir)' in her2_block
    assert "'train': 'train.json', 'val': 'val.json', 'test': 'test.json'" in her2_block


def test_her2_is_registered_in_the_cli():
    main_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'main.py')
    with open(main_path, 'rt', encoding='utf-8') as f:
        source = f.read()
    assert "'her2': HER2Dataset," in source
    assert "parser.add_argument('--her2_dir', type=str)" in source
    assert "her2_dir='./data/her2_512p/'," in source
