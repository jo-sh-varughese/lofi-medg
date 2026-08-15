'''
Tests for the shuffled-image control (main.py --shuffle_images).

RESULTS_her2.md section 6 makes this control a gate on every other number: if it
scores close to the real evaluation, the model is ignoring the image. A control
that quietly left some samples correctly paired would understate the collapse,
so the derangement property is asserted here rather than assumed.

Imported by source extraction so these run without torch, matching the rest of
the suite.
'''

import ast
import os

import pytest

MAIN_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'main.py')


def load_shuffle_images():
    '''Pull shuffle_images out of main.py verbatim; main.py itself needs torch.'''
    with open(MAIN_PATH, 'rt', encoding='utf-8') as f:
        tree = ast.parse(f.read())
    namespace = {'random': __import__('random')}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'shuffle_images':
            exec(ast.unparse(node), namespace)
            return namespace['shuffle_images']
    raise AssertionError('shuffle_images not found in main.py')


class Stub:
    def __init__(self, n, with_metas=False):
        self.imgs = [f'/data/img_{i}.png' for i in range(n)]
        self.qas = [(f'label_{i}', f'[[{i},{i},{i},{i}]]') for i in range(n)]
        self.metas = ([{'image_path': p, 'content_type': f'type_{i % 2}'}
                       for i, p in enumerate(self.imgs)] if with_metas else None)


@pytest.fixture
def shuffle_images():
    return load_shuffle_images()


class TestDerangement:
    @pytest.mark.parametrize('n', [2, 3, 5, 50, 257])
    def test_no_sample_keeps_its_own_image(self, shuffle_images, n):
        dataset = Stub(n)
        original = list(dataset.imgs)
        shuffle_images(dataset, seed=42)
        assert all(a != b for a, b in zip(original, dataset.imgs)), 'a sample kept its own image'

    def test_a_single_sample_is_refused_rather_than_silently_unpaired(self, shuffle_images):
        with pytest.raises(ValueError, match='at least 2 samples'):
            shuffle_images(Stub(1), seed=42)

    def test_every_image_is_still_used_exactly_once(self, shuffle_images):
        dataset = Stub(64)
        original = list(dataset.imgs)
        shuffle_images(dataset, seed=7)
        assert sorted(dataset.imgs) == sorted(original)


class TestTargetsUntouched:
    def test_questions_and_boxes_are_unchanged(self, shuffle_images):
        dataset = Stub(32)
        original_qas = list(dataset.qas)
        shuffle_images(dataset, seed=1)
        assert dataset.qas == original_qas, 'the control must move images, not targets'

    def test_content_type_stays_with_the_target(self, shuffle_images):
        # so the per-track breakdown splits the control exactly as it splits the
        # real run; a control that reshuffled content_type would not be comparable
        dataset = Stub(32, with_metas=True)
        original_types = [m['content_type'] for m in dataset.metas]
        shuffle_images(dataset, seed=1)
        assert [m['content_type'] for m in dataset.metas] == original_types

    def test_recorded_image_path_follows_the_new_pairing(self, shuffle_images):
        dataset = Stub(32, with_metas=True)
        shuffle_images(dataset, seed=1)
        assert [m['image_path'] for m in dataset.metas] == dataset.imgs


class TestDeterminism:
    def test_same_seed_gives_the_same_pairing(self, shuffle_images):
        a, b = Stub(40), Stub(40)
        shuffle_images(a, seed=42)
        shuffle_images(b, seed=42)
        assert a.imgs == b.imgs

    def test_different_seeds_give_different_pairings(self, shuffle_images):
        a, b = Stub(40), Stub(40)
        shuffle_images(a, seed=42)
        shuffle_images(b, seed=43)
        assert a.imgs != b.imgs


class TestResultFileSeparation:
    def test_control_writes_to_a_distinct_filename(self):
        # the control reuses the same checkpoint, so without a suffix it would
        # overwrite the real evaluation's csv
        with open(MAIN_PATH, 'rt', encoding='utf-8') as f:
            source = f.read()
        assert "control_suffix = '_shuffled' if args.shuffle_images else ''" in source
        assert 'ground_{name}{control_suffix}.csv' in source
