import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from lofi_utils.data import fuse_boxes


#######
# Box
#######
def get_boxes_from_mask(mask, area_threshold=0):
    '''
    mask: (H, W) binary mask
    return: [[x0, y0, x1, y1], ...]  # normalized box list
    '''
    H, W = mask.shape

    # connectedComponentsWithStats expects uint8
    mask = mask.astype(np.uint8, copy=False)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return []

    stats = stats[1:]  # skip background

    if area_threshold > 0:
        area_threshold_pixel_level = int((W * area_threshold) * (H * area_threshold))
        stats = stats[stats[:, cv2.CC_STAT_AREA] >= area_threshold_pixel_level]

    if stats.shape[0] == 0:
        return []

    x = stats[:, cv2.CC_STAT_LEFT].astype(np.float32)
    y = stats[:, cv2.CC_STAT_TOP].astype(np.float32)
    bw = stats[:, cv2.CC_STAT_WIDTH].astype(np.float32)
    bh = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float32)

    x0 = np.clip(x / W, 0.0, 1.0)
    y0 = np.clip(y / H, 0.0, 1.0)
    x1 = np.clip((x + bw) / W, 0.0, 1.0)
    y1 = np.clip((y + bh) / H, 0.0, 1.0)

    return np.stack([x0, y0, x1, y1], axis=1).tolist()


def remove_inner_boxes(boxes):
    inside = lambda a, b: a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]
    result = []
    for i, b in enumerate(boxes):
        if not any(inside(b, o) for j, o in enumerate(boxes) if i != j):
            result.append(b)
    return result


def seg_slice_to_boxes(seg_slice, area_threshold=0.04, iou_thr=0.5):
    boxes = get_boxes_from_mask(seg_slice, area_threshold=area_threshold)  # (clip 0~1)
    boxes = remove_inner_boxes(boxes)  # remove inner boxes
    boxes = fuse_boxes(boxes, iou_thr=iou_thr)  # fuse boxes
    boxes.sort(key=lambda b: b[0])  # sort boxes
    return boxes


#########
# Image
#########
def resize_with_padding(image, size):
    W, H = image.size
    scale = size / max(W, H)  # longer to size
    nw, nh = int(round(W * scale)), int(round(H * scale))
    if (nw, nh) != (W, H):
        image = image.resize((nw, nh), resample=3)  # 3 = Image.LANCZOS
    canvas = Image.new('RGB', (size, size), (0, 0, 0))
    canvas.paste(image, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def save_jpg(image, save_path):
    save_path = os.path.splitext(save_path)[0] + '.jpg'  # always jpg
    image.save(
        save_path,
        'JPEG',
        quality=100,  # max visual quality
        subsampling=0,  # preserve color detail
        optimize=True,  # compress efficiently
        progressive=True  # better progressive loading
    )


#######
# Etc
#######
def check_image_using_pixel(arr, threshold=0.9):
    assert len(arr.shape) == 2
    values, counts = np.unique(arr, return_counts=True)
    return counts.max() / arr.size >= threshold
