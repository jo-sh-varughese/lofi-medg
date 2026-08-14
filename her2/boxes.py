'''
Box construction for the HER2 dataset.

Every helper here returns boxes in the exact space the LoFi loader expects:
integer pixel coordinates [x0, y0, x1, y1] on the letterboxed target_size canvas,
sorted by x0. `lofi_utils/dataset/det.py` then divides by target_annotated_size
and multiplies by 1000 to reach the 0-1000 bins the decoder emits as text.
'''

import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from lofi_utils.misc import convert_pad_space
from tools.preprocess_utils import seg_slice_to_boxes

# The decoder writes boxes as literal text tokens, so a sample with many boxes
# eats the caption budget. MedG samples are typically a handful of regions.
MAX_BOXES = 8

# Measured against the gemma-3-270m-it tokenizer over every template in
# her2/captions.py, using worst-case four-digit coordinates and the prompt built
# by lofi_utils/dataset/base.py. Reproduce with tools/check_her2_token_budget.py.
#
#     tokens(n boxes) = TOKEN_BASE + TOKEN_PER_BOX * n     (+ multimodal_tokens)
#
# main.py adds multimodal_tokens to decoder_max_length, so the usable budget is
# (decoder_max_length + multimodal_tokens). Exceeding it makes pad_or_truncate cut
# the tail off the target -- including the closing fence and <end_of_turn> -- which
# corrupts training silently rather than raising.
TOKEN_BASE = 168
TOKEN_PER_BOX = 20


def max_boxes_for(decoder_max_length=150, multimodal_tokens=128):
    '''
    Largest number of boxes per sample that fits the decoder budget intact.

    150 + 128 -> 5 boxes;  200 + 128 -> 8 boxes.
    '''
    budget = decoder_max_length + multimodal_tokens
    return max(0, (budget - TOKEN_BASE) // TOKEN_PER_BOX)


def required_decoder_max_length(max_boxes=MAX_BOXES, multimodal_tokens=128):
    '''
    Smallest --decoder_max_length that keeps `max_boxes` boxes intact.
    '''
    return max(0, TOKEN_BASE + TOKEN_PER_BOX * max_boxes - multimodal_tokens)

# Discard slivers: a box narrower than this fraction of the image is noise at
# 512p and cannot be matched at IoU 0.5 anyway.
MIN_BOX_SIDE = 0.02


def clean_mask(mask, close_px=5, open_px=3):
    '''
    Morphologically tidy a raw threshold mask: close pinholes inside stained
    membranes, then drop isolated speckle.

    mask: (H, W) uint8
    return: (H, W) uint8
    '''
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if close_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if open_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def filter_boxes(boxes, min_side=MIN_BOX_SIDE, max_boxes=MAX_BOXES):
    '''
    boxes: normalized [[x0, y0, x1, y1], ...]
    return: normalized boxes, largest-area first truncated to max_boxes, sorted by x0
    '''
    kept = [b for b in boxes if (b[2] - b[0]) >= min_side and (b[3] - b[1]) >= min_side]
    if len(kept) > max_boxes:
        kept = sorted(kept, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:max_boxes]
    kept.sort(key=lambda b: b[0])
    return kept


def mask_to_boxes(mask, area_threshold=0.05, iou_thr=0.5, min_side=MIN_BOX_SIDE, max_boxes=MAX_BOXES):
    '''
    Connected components of a binary mask -> normalized boxes.

    Uses the repository's own `seg_slice_to_boxes` so HER2 regions are derived
    exactly the way SegTHOR and TotalSegmentator regions are.

    mask: (H, W) binary
    return: normalized [[x0, y0, x1, y1], ...]
    '''
    boxes = seg_slice_to_boxes(mask, area_threshold=area_threshold, iou_thr=iou_thr)
    return filter_boxes(boxes, min_side=min_side, max_boxes=max_boxes)


def polygons_to_boxes(polygons, width, height, min_side=MIN_BOX_SIDE, max_boxes=MAX_BOXES):
    '''
    Polygon vertex lists -> normalized boxes, clipped to the image.

    polygons: [[(x, y), ...], ...] in pixel coordinates of a (width, height) image
    return: normalized [[x0, y0, x1, y1], ...]
    '''
    boxes = []
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        if len(points) < 3:
            continue
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        box = [
            float(np.clip(x0 / width, 0.0, 1.0)),
            float(np.clip(y0 / height, 0.0, 1.0)),
            float(np.clip(x1 / width, 0.0, 1.0)),
            float(np.clip(y1 / height, 0.0, 1.0)),
        ]
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
    return filter_boxes(boxes, min_side=min_side, max_boxes=max_boxes)


def to_padded_pixels(boxes, ori_size, target_size):
    '''
    Normalized original-image boxes -> integer pixel boxes on the padded
    target_size canvas, matching `tools/preprocess_segthor.py`.

    boxes: normalized [[x0, y0, x1, y1], ...]
    ori_size: (width, height) of the image before letterboxing
    return: [[int, int, int, int], ...] sorted by x0
    '''
    if len(boxes) == 0:
        return []
    padded = convert_pad_space(boxes, ori_size, target_size)
    out = []
    for box in padded:
        x0, y0, x1, y1 = [int(round(v * target_size)) for v in box]
        x0 = max(0, min(x0, target_size))
        y0 = max(0, min(y0, target_size))
        x1 = max(0, min(x1, target_size))
        y1 = max(0, min(y1, target_size))
        if x1 > x0 and y1 > y0:
            out.append([x0, y0, x1, y1])
    out.sort(key=lambda b: b[0])
    return out


def whole_tissue_box(mask, ori_size, target_size, fallback_full=True):
    '''
    Single box around all tissue in the image. Used for HER2 0 samples, where
    the clinically correct statement is that no membranous staining is present
    anywhere in the tissue -- there is no sub-region to localise.

    return: [[int, int, int, int]] or []
    '''
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        if not fallback_full:
            return []
        box = [0.0, 0.0, 1.0, 1.0]
    else:
        box = [
            float(xs.min() / width),
            float(ys.min() / height),
            float((xs.max() + 1) / width),
            float((ys.max() + 1) / height),
        ]
    return to_padded_pixels([box], ori_size, target_size)
