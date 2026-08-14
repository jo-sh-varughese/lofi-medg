import numpy as np

from her2 import boxes as her2_boxes
from her2 import stain
from tests.conftest import DAB_BROWN, GLASS, HEMATOXYLIN_BLUE, make_ihc_patch


def test_deconvolution_separates_brown_from_blue():
    field = np.zeros((8, 8, 3), dtype=np.uint8)
    field[:, :4] = DAB_BROWN
    field[:, 4:] = HEMATOXYLIN_BLUE
    hematoxylin, dab = stain.separate_stains(field)

    assert dab[:, :4].mean() > dab[:, 4:].mean()
    assert hematoxylin[:, 4:].mean() > hematoxylin[:, :4].mean()


def test_glass_has_almost_no_stain():
    glass = np.full((8, 8, 3), GLASS, dtype=np.uint8)
    _, dab = stain.separate_stains(glass)
    assert dab.max() < stain.DAB_TISSUE_THRESHOLD


def test_dab_mask_tracks_grade():
    fractions = [stain.dab_fraction(make_ihc_patch(grade)) for grade in ['0', '1+', '3+']]
    assert fractions[0] < fractions[1] < fractions[2]


def test_strong_threshold_selects_only_dark_chromogen():
    strong = stain.dab_mask(make_ihc_patch('3+'), threshold=stain.DAB_STRONG_THRESHOLD)
    faint = stain.dab_mask(make_ihc_patch('1+'), threshold=stain.DAB_STRONG_THRESHOLD)
    assert strong.sum() > 0
    assert faint.sum() < strong.sum()


def test_mask_to_boxes_finds_both_blobs():
    mask = stain.dab_mask(make_ihc_patch('3+'))
    mask = her2_boxes.clean_mask(mask)
    found = her2_boxes.mask_to_boxes(mask, area_threshold=0.06)

    assert len(found) == 2
    for box in found:
        assert 0.0 <= box[0] < box[2] <= 1.0
        assert 0.0 <= box[1] < box[3] <= 1.0
    assert found[0][0] <= found[1][0]  # sorted by x0


def test_to_padded_pixels_is_integral_and_in_range():
    padded = her2_boxes.to_padded_pixels([[0.1, 0.2, 0.5, 0.6]], (256, 256), 512)
    assert padded == [[51, 102, 256, 307]]
    assert all(isinstance(v, int) for v in padded[0])


def test_to_padded_pixels_accounts_for_letterbox_padding():
    # a 2:1 landscape image is padded top and bottom; a full-width box must land
    # in the vertical middle band of the canvas, not at y=0..512
    padded = her2_boxes.to_padded_pixels([[0.0, 0.0, 1.0, 1.0]], (512, 256), 512)
    x0, y0, x1, y1 = padded[0]
    assert (x0, x1) == (0, 512)
    assert y0 == 128 and y1 == 384


def test_filter_boxes_drops_slivers_and_caps_count():
    sliver = [0.5, 0.5, 0.505, 0.9]
    assert her2_boxes.filter_boxes([sliver]) == []

    many = [[i / 100, 0.1, i / 100 + 0.1, 0.9] for i in range(20)]
    assert len(her2_boxes.filter_boxes(many, max_boxes=8)) == 8


def test_polygons_to_boxes_uses_polygon_extent():
    polygon = [(10, 20), (110, 20), (110, 220), (10, 220)]
    found = her2_boxes.polygons_to_boxes([polygon], 200, 400)
    assert len(found) == 1
    assert found[0] == [0.05, 0.05, 0.55, 0.55]


def test_whole_tissue_box_wraps_all_tissue():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 30:60] = 1
    padded = her2_boxes.whole_tissue_box(mask, (100, 100), 512)
    # x: 30/100 and 60/100 of 512 -> 154, 307;  y: 20/100 and 80/100 -> 102, 410
    assert padded == [[154, 102, 307, 410]]
