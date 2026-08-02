import os
import random
import re

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # make CUDA kernels deterministic when possible.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def seed_worker(*args, **kwargs):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_target_annotated_size(dataset_dir, default=512):
    match = re.search(r'_(\d+)p(?:_|$)', os.path.basename(os.path.normpath(dataset_dir)).lower())
    target_annotated_size = int(match.group(1)) if match else default
    print(f'target_annotated_size={target_annotated_size}')
    return target_annotated_size


def _assert_normalized_boxes(b, eps=1e-6):
    if b.size and not (b.min() >= -eps and b.max() <= 1 + eps):
        raise AssertionError(f'boxes out of [0,1]: min={b.min():.4f}, max={b.max():.4f}')


def convert_ori_space(boxes, ori_size, size):
    # boxes should be between 0 and 1
    # for ori_size, the exact values are not important; only the aspect ratio (W:H) matters.
    # for size, the exact value is usually not important because scale terms cancel after normalization.
    # pixel-space mapping based on integer resized shape (nw, nh) and integer centered padding (px, py).
    W, H = ori_size
    s = size / max(W, H)
    nw, nh = int(round(W * s)), int(round(H * s))

    b = np.asarray(boxes, np.float32)
    if b.size == 0:
        return []

    _assert_normalized_boxes(b)

    # normalized pad-space -> pixel pad-space
    b = b * size

    # padding offsets
    px, py = (size - nw) // 2, (size - nh) // 2

    # remove padding, map back to original normalized space
    b[:, [0, 2]] = np.clip(b[:, [0, 2]] - px, 0, nw) / (s * W)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]] - py, 0, nh) / (s * H)

    return b.tolist()


def convert_pad_space(boxes, ori_size, size):
    # boxes should be between 0 and 1
    # for ori_size, the exact values are not important; only the aspect ratio (W:H) matters.
    # for size, the exact value is usually not important because scale terms cancel after normalization.
    # pixel-space mapping based on integer resized shape (nw, nh) and integer centered padding (px, py).
    W, H = ori_size
    s = size / max(W, H)
    nw, nh = int(round(W * s)), int(round(H * s))

    b = np.asarray(boxes, np.float32)
    if b.size == 0:
        return []

    _assert_normalized_boxes(b)

    # padding offsets
    px, py = (size - nw) // 2, (size - nh) // 2

    # original normalized -> resized pixel -> add padding -> normalized pad-space
    b[:, [0, 2]] = (b[:, [0, 2]] * W * s + px) / size
    b[:, [1, 3]] = (b[:, [1, 3]] * H * s + py) / size

    return b.tolist()
