'''
CAMELYON track coverage that does not need OpenSlide: annotation parsing, tile
window clipping, and slide-to-patient split keying.
'''

import numpy as np

from her2 import boxes as her2_boxes
from her2 import wsi
from tools import preprocess_camelyon

ASAP_XML = '''<?xml version="1.0"?>
<ASAP_Annotations>
  <Annotations>
    <Annotation Name="Annotation 0" Type="Polygon" PartOfGroup="metastases" Color="#F4FA58">
      <Coordinates>
        <Coordinate Order="0" X="1000" Y="2000" />
        <Coordinate Order="1" X="1400" Y="2000" />
        <Coordinate Order="2" X="1400" Y="2400" />
        <Coordinate Order="3" X="1000" Y="2400" />
      </Coordinates>
    </Annotation>
    <Annotation Name="Annotation 1" Type="Polygon" PartOfGroup="normal" Color="#000000">
      <Coordinates>
        <Coordinate Order="0" X="9000" Y="9000" />
        <Coordinate Order="1" X="9100" Y="9000" />
        <Coordinate Order="2" X="9100" Y="9100" />
      </Coordinates>
    </Annotation>
    <Annotation Name="Annotation 2" Type="Dot" PartOfGroup="metastases">
      <Coordinates>
        <Coordinate Order="0" X="5" Y="5" />
      </Coordinates>
    </Annotation>
  </Annotations>
</ASAP_Annotations>
'''


def write_xml(tmp_path):
    path = tmp_path / 'tumor_001.xml'
    path.write_text(ASAP_XML, encoding='utf-8')
    return str(path)


def test_parses_polygon_groups_and_skips_degenerate_shapes(tmp_path):
    annotations = wsi.read_asap_annotations(write_xml(tmp_path))

    # the two-point 'Dot' annotation has fewer than 3 vertices and is dropped
    assert len(annotations) == 2
    groups = [group for group, _ in annotations]
    assert groups == ['metastases', 'normal']
    assert len(annotations[0][1]) == 4


def test_find_annotation_file_matches_slide_stem(tmp_path):
    write_xml(tmp_path)
    assert wsi.find_annotation_file(str(tmp_path), '/data/slides/Tumor_001.tif') is not None
    assert wsi.find_annotation_file(str(tmp_path), '/data/slides/tumor_002.tif') is None
    assert wsi.find_annotation_file('', '/data/slides/tumor_001.tif') is None


def test_polygons_clip_into_tile_coordinates(tmp_path):
    annotations = wsi.read_asap_annotations(write_xml(tmp_path))

    # a 1024-pixel tile at level 0 starting at (1000, 2000) contains the lesion
    inside = wsi.polygons_in_window(annotations, 1000, 2000, 1024, 1024, downsample=1.0)
    assert len(inside) == 1
    group, points = inside[0]
    assert group == 'metastases'

    array = np.asarray(points)
    assert array.min() >= 0 and array.max() <= 1023
    assert array[0].tolist() == [0.0, 0.0]  # polygon origin sits at the tile corner


def test_polygons_outside_the_window_are_excluded(tmp_path):
    annotations = wsi.read_asap_annotations(write_xml(tmp_path))
    assert wsi.polygons_in_window(annotations, 0, 0, 512, 512, downsample=1.0) == []


def test_clipped_polygon_becomes_a_valid_padded_box(tmp_path):
    annotations = wsi.read_asap_annotations(write_xml(tmp_path))
    inside = wsi.polygons_in_window(annotations, 1000, 2000, 1024, 1024, downsample=1.0)

    normalized = her2_boxes.polygons_to_boxes([points for _, points in inside], 1024, 1024)
    padded = her2_boxes.to_padded_pixels(normalized, (1024, 1024), 512)

    assert len(padded) == 1
    x0, y0, x1, y1 = padded[0]
    assert 0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 512
    assert all(isinstance(v, int) for v in padded[0])


def test_downsampled_window_scales_coordinates(tmp_path):
    annotations = wsi.read_asap_annotations(write_xml(tmp_path))
    # a level-2 tile (downsample 4) covering 4096 level-0 pixels
    inside = wsi.polygons_in_window(annotations, 1000, 2000, 4096, 4096, downsample=4.0)
    array = np.asarray(inside[0][1])
    assert array.max() <= 1024  # 400 level-0 pixels / 4 = 100 tile pixels
    assert np.isclose(array[:, 0].max(), 100.0)


def test_slide_id_groups_camelyon17_nodes_by_patient():
    assert preprocess_camelyon.slide_id_of('/d/patient_004_node_4.tif') == 'patient_004'
    assert preprocess_camelyon.slide_id_of('/d/patient_004_node_1.tif') == 'patient_004'
    assert preprocess_camelyon.slide_id_of('/d/tumor_001.tif') == 'tumor_001'


def test_tissue_ratio_separates_glass_from_stained_tissue():
    glass = np.full((32, 32, 3), 245, dtype=np.uint8)
    tissue = np.zeros((32, 32, 3), dtype=np.uint8)
    tissue[:, :] = (150, 90, 160)

    assert wsi.tissue_ratio(glass) < 0.05
    assert wsi.tissue_ratio(tissue) > 0.95
