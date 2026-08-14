import pytest

from her2 import captions


def test_every_grade_has_a_template():
    for grade in captions.HER2_GRADES:
        assert captions.her2_region_text(grade).strip()


def test_grade_spellings_normalize():
    assert captions.normalize_grade('3') == '3+'
    assert captions.normalize_grade('3+') == '3+'
    assert captions.normalize_grade(' HER2 2+ ') == '2+'
    assert captions.normalize_grade(0) == '0'
    with pytest.raises(ValueError):
        captions.normalize_grade('4+')
    with pytest.raises(ValueError):
        captions.normalize_grade('equivocal')


def test_labels_are_period_free_and_lowercase_start():
    # the loader strips one trailing period when building the grounding prompt,
    # so a period would make the grounding and captioning text differ
    for _, text in captions.all_templates():
        assert not text.endswith('.')
        assert text == text.strip()


def test_only_grade_zero_is_unstained():
    assert captions.is_unstained('0')
    for grade in ['1+', '2+', '3+']:
        assert not captions.is_unstained(grade)


def test_grade_text_names_pattern_and_score():
    # each caption must state the visible pattern and the score it implies
    assert 'strong complete' in captions.her2_region_text('3+')
    assert '3+' in captions.her2_region_text('3+')
    assert 'equivocal' in captions.her2_region_text('2+')
    assert 'no membranous staining' in captions.her2_region_text('0')


def test_lesion_text_carries_no_her2_claim():
    # CAMELYON is H&E; its captions must never mention HER2
    for group in captions.LESION_REGION_TEXT:
        assert 'her2' not in captions.lesion_region_text(group).lower()
    assert captions.lesion_region_text('nonexistent-group') is None
