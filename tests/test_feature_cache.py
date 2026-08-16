'''
Tests for the frozen-encoder feature cache.

These cover the parts that decide correctness -- keying, staleness detection
and round-tripping -- all of which are pure Python/numpy and run without torch,
a GPU or any downloaded data. The encoder pass itself (tools/precompute_features.py)
is NOT covered here; it needs the real checkpoints.
'''

import json
import os

import numpy as np
import pytest

from lofi_utils.feature_cache import (
    FeatureCache, build_manifest, cache_file_path, cache_key, check_manifest,
    estimate_cache_bytes, read_manifest, save_feature, unique_images, write_manifest,
)


def _manifest(**overrides):
    base = build_manifest(
        model_name='siglip2-so400m-patch16-512-lofi-medg',
        resume_path='',
        image_size=512,
        pool2x2=True,
        dtype='float16',
        feature_shape=(256, 1152),
    )
    base.update(overrides)
    return base


class TestCacheKey:
    def test_relative_and_absolute_paths_agree(self, tmp_path):
        target = tmp_path / 'a.png'
        target.write_bytes(b'')
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            assert cache_key('a.png') == cache_key(str(target))
        finally:
            os.chdir(cwd)

    def test_different_images_get_different_keys(self):
        assert cache_key('/data/a.png') != cache_key('/data/b.png')

    def test_path_is_sharded_by_first_two_characters(self):
        path = cache_file_path('/cache', '/data/a.png')
        key = cache_key('/data/a.png')
        assert os.path.basename(path) == f'{key}.npy'
        assert os.path.basename(os.path.dirname(path)) == key[:2]


class TestManifestValidation:
    def test_identical_manifests_have_no_problems(self):
        assert check_manifest(_manifest(), _manifest()) == []

    @pytest.mark.parametrize('field,value', [
        ('model_name', 'siglip2-so400m-patch16-512'),
        ('image_size', 256),
        ('pool2x2', False),
        ('dtype', 'float32'),
    ])
    def test_each_field_mismatch_is_reported(self, field, value):
        problems = check_manifest(_manifest(), _manifest(**{field: value}))
        assert len(problems) == 1
        assert field in problems[0]

    def test_cache_from_an_older_encoder_build_is_rejected(self):
        '''
        A cache written before encoder_build existed came from the LoRA-module
        encoder, which this version replaced with merged weights. Every other
        field still matches, so this is the only thing standing between a stale
        cache and a silently wrong run.
        '''
        stale = _manifest()
        del stale['encoder_build']
        problems = check_manifest(_manifest(), stale)
        assert len(problems) == 1
        assert 'encoder_build' in problems[0]

    def test_build_manifest_stamps_the_encoder_build(self):
        assert build_manifest('m', '', 512, True, 'float16', (256, 1152))['encoder_build']

    def test_different_resume_checkpoint_is_reported(self, tmp_path):
        ckpt = tmp_path / 'last.pt'
        ckpt.write_bytes(b'x' * 100)
        with_ckpt = build_manifest('m', str(ckpt), 512, True, 'float16', (256, 1152))
        without = _manifest(model_name='m')
        problems = check_manifest(with_ckpt, without)
        assert any('resume checkpoint differs' in p for p in problems)

    def test_unsupported_dtype_is_rejected_at_build_time(self):
        with pytest.raises(ValueError, match='unsupported cache dtype'):
            build_manifest('m', '', 512, True, 'bfloat16', (256, 1152))


class TestRoundTrip:
    def test_saved_feature_loads_back_unchanged(self, tmp_path):
        cache_dir = str(tmp_path / 'cache')
        feature = np.arange(256 * 8, dtype='float16').reshape(256, 8)
        save_feature(cache_dir, '/data/a.png', feature)
        write_manifest(cache_dir, _manifest(feature_shape=[256, 8]))

        cache = FeatureCache(cache_dir)
        np.testing.assert_array_equal(cache.load('/data/a.png'), feature)

    def test_no_temporary_files_survive_a_save(self, tmp_path):
        cache_dir = str(tmp_path / 'cache')
        save_feature(cache_dir, '/data/a.png', np.zeros((4, 4), dtype='float16'))
        leftovers = [p for _, _, files in os.walk(cache_dir) for p in files if '.tmp' in p]
        assert leftovers == []

    def test_missing_manifest_is_an_error_not_an_empty_cache(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='precompute_features'):
            FeatureCache(str(tmp_path))

    def test_mismatched_manifest_refuses_to_load(self, tmp_path):
        cache_dir = str(tmp_path / 'cache')
        write_manifest(cache_dir, _manifest(model_name='some-other-encoder'))
        with pytest.raises(ValueError, match='does not match this run'):
            FeatureCache(cache_dir, expected_manifest=_manifest())

    def test_absent_image_raises_rather_than_silently_recomputing(self, tmp_path):
        cache_dir = str(tmp_path / 'cache')
        write_manifest(cache_dir, _manifest())
        cache = FeatureCache(cache_dir)
        with pytest.raises(KeyError, match='not in the feature cache'):
            cache.load('/data/never_cached.png')

    def test_missing_reports_only_uncached_paths_once(self, tmp_path):
        cache_dir = str(tmp_path / 'cache')
        save_feature(cache_dir, '/data/a.png', np.zeros((4, 4), dtype='float16'))
        write_manifest(cache_dir, _manifest())
        cache = FeatureCache(cache_dir)
        assert cache.missing(['/data/a.png', '/data/b.png', '/data/b.png']) == ['/data/b.png']


class TestDeduplication:
    def test_repeated_images_are_encoded_once(self):
        class Stub:
            imgs = ['/a.png', '/b.png', '/a.png', '/c.png', '/b.png']

        assert unique_images(Stub()) == ['/a.png', '/b.png', '/c.png']

    def test_size_estimate_matches_actual_bytes(self):
        # 100 images of 256x1152 float16 == the real on-disk payload size
        assert estimate_cache_bytes(100, (256, 1152), 'float16') == 100 * 256 * 1152 * 2
        assert estimate_cache_bytes(100, (256, 1152), 'float32') == 100 * 256 * 1152 * 4


class TestManifestIO:
    def test_manifest_round_trips_through_disk(self, tmp_path):
        cache_dir = str(tmp_path / 'cache')
        manifest = _manifest()
        write_manifest(cache_dir, manifest)
        assert read_manifest(cache_dir) == manifest

    def test_manifest_is_readable_json(self, tmp_path):
        cache_dir = str(tmp_path / 'cache')
        write_manifest(cache_dir, _manifest())
        with open(os.path.join(cache_dir, 'manifest.json'), 'rt', encoding='utf-8') as f:
            assert json.load(f)['pool2x2'] is True

    def test_absent_directory_reads_as_none(self, tmp_path):
        assert read_manifest(str(tmp_path / 'nope')) is None
