'''
Whole-slide image access and ASAP annotation parsing.

LoFi has no notion of a whole-slide image: every dataset in the repository is a
set of independent 2D frames, and 3D volumes are handled by exporting slices
(see `tools/preprocess_segthor.py`). This module follows the same convention by
exporting a WSI as independent tiles, each of which becomes one standalone 512p
image with its own boxes.

OpenSlide is imported lazily so the format tests, the caption templates and the
manifest validator all run in an environment without the OpenSlide binaries.
'''

import os
import xml.etree.ElementTree as ET

import numpy as np

# Tiles are cut at this microns-per-pixel by default. HER2 scoring is a
# membrane-level judgement, so we stay near 10x-20x rather than at thumbnail
# scale, and one tile covers roughly 0.5-1 mm of tissue.
DEFAULT_MPP = 1.0

# Fraction of a tile that must be tissue for the tile to be exported.
DEFAULT_TISSUE_RATIO = 0.35


def load_openslide():
    try:
        import openslide
    except ImportError as error:
        raise ImportError(
            'openslide-python is required for whole-slide input. '
            'Install the OpenSlide binaries plus `pip install openslide-python`, '
            'or use a patch-level source such as BCI that needs no WSI reader.'
        ) from error
    return openslide


def open_slide(slide_path):
    openslide = load_openslide()
    return openslide.OpenSlide(slide_path)


def slide_mpp(slide, default=0.25):
    '''
    Microns per pixel at level 0, from the slide properties. Falls back to
    `default` (a typical 40x scan) when the vendor metadata is absent.
    '''
    for key in ('openslide.mpp-x', 'aperio.MPP', 'hamamatsu.XResolution'):
        value = slide.properties.get(key)
        if value:
            try:
                return float(value)
            except ValueError:
                continue
    return default


def pick_level(slide, target_mpp=DEFAULT_MPP, default_mpp=0.25):
    '''
    Best pyramid level at or above the requested resolution.

    return: (level, downsample, mpp_at_level)
    '''
    base_mpp = slide_mpp(slide, default=default_mpp)
    wanted = target_mpp / base_mpp  # desired downsample factor
    level = 0
    for index, downsample in enumerate(slide.level_downsamples):
        if downsample <= wanted * 1.5:
            level = index
    downsample = slide.level_downsamples[level]
    return level, downsample, base_mpp * downsample


def tissue_ratio(tile_rgb, saturation_threshold=20, value_threshold=235):
    '''
    Fraction of a tile that is tissue rather than blank glass. Uses the standard
    saturation heuristic: glass is bright and colourless, tissue is not.

    tile_rgb: (H, W, 3) uint8
    '''
    array = np.asarray(tile_rgb, dtype=np.float32)
    maximum = array.max(axis=2)
    minimum = array.min(axis=2)
    saturation = np.where(maximum > 0, (maximum - minimum) / np.maximum(maximum, 1e-6) * 255.0, 0.0)
    tissue = (saturation > saturation_threshold) & (maximum < value_threshold)
    return float(tissue.mean())


def iter_tiles(slide, tile_size=1024, target_mpp=DEFAULT_MPP, stride=None, tissue_ratio_min=DEFAULT_TISSUE_RATIO, limit=None):
    '''
    Yield tissue-bearing tiles from a slide.

    tile_size: tile edge in pixels at the chosen pyramid level
    yield: (level0_x, level0_y, level, downsample, PIL.Image RGB tile)
    '''
    level, downsample, _ = pick_level(slide, target_mpp=target_mpp)
    stride = tile_size if stride is None else stride
    level_width, level_height = slide.level_dimensions[level]

    produced = 0
    for level_y in range(0, max(1, level_height - tile_size + 1), stride):
        for level_x in range(0, max(1, level_width - tile_size + 1), stride):
            base_x = int(level_x * downsample)
            base_y = int(level_y * downsample)
            tile = slide.read_region((base_x, base_y), level, (tile_size, tile_size)).convert('RGB')
            if tissue_ratio(np.asarray(tile)) < tissue_ratio_min:
                continue
            yield base_x, base_y, level, downsample, tile
            produced += 1
            if limit is not None and produced >= limit:
                return


def read_asap_annotations(xml_path):
    '''
    Parse a CAMELYON16/17 ASAP annotation file.

    These are real polygons drawn by pathologists over metastatic regions; they
    are the only genuinely region-level annotations in this project.

    return: [(group_name, [(x, y), ...]), ...] in level-0 pixel coordinates
    '''
    tree = ET.parse(xml_path)
    annotations = []
    for node in tree.iter('Annotation'):
        group = node.get('PartOfGroup') or node.get('Name') or ''
        group = group.split('_')[0].strip().lower()
        points = []
        for coordinate in node.iter('Coordinate'):
            try:
                points.append((float(coordinate.get('X')), float(coordinate.get('Y'))))
            except (TypeError, ValueError):
                continue
        if len(points) >= 3:
            annotations.append((group, points))
    return annotations


def polygons_in_window(annotations, x, y, width, height, downsample=1.0, min_overlap=0.02):
    '''
    Clip level-0 polygons into a tile window and re-express them in tile pixel
    coordinates.

    x, y, width, height: tile window in level-0 pixels (width/height already
        multiplied by downsample by the caller)
    return: [(group_name, [(tile_x, tile_y), ...]), ...]
    '''
    inside = []
    for group, points in annotations:
        array = np.asarray(points, dtype=np.float64)
        within = (
            (array[:, 0] >= x) & (array[:, 0] < x + width) &
            (array[:, 1] >= y) & (array[:, 1] < y + height)
        )
        if within.mean() < min_overlap:
            continue
        clipped = array.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], x, x + width - 1)
        clipped[:, 1] = np.clip(clipped[:, 1], y, y + height - 1)
        clipped[:, 0] = (clipped[:, 0] - x) / downsample
        clipped[:, 1] = (clipped[:, 1] - y) / downsample
        inside.append((group, [tuple(point) for point in clipped]))
    return inside


def find_annotation_file(annotation_dir, slide_path):
    '''
    Locate the XML that goes with a slide, tolerating the case differences between
    CAMELYON16 (`Tumor_001.xml`) and CAMELYON17 (`patient_004_node_4.xml`).
    '''
    if not annotation_dir or not os.path.isdir(annotation_dir):
        return None
    stem = os.path.splitext(os.path.basename(slide_path))[0].lower()
    for filename in os.listdir(annotation_dir):
        if not filename.lower().endswith('.xml'):
            continue
        if os.path.splitext(filename)[0].lower() == stem:
            return os.path.join(annotation_dir, filename)
    return None
