'''
Colour deconvolution for H-DAB immunohistochemistry.

Implements the Ruifrok & Johnston (2001) optical-density model, which is the
standard way to separate the haematoxylin (nuclear, blue) and DAB (chromogen,
brown) components of an IHC slide.

Why this is a legitimate signal for HER2, and only on IHC:
    HER2 immunohistochemistry visualises membrane HER2 protein with a DAB
    chromogen. The DAB concentration map therefore measures the thing the HER2
    score is defined on. Running the same deconvolution on an H&E slide would
    measure haematoxylin/eosin bleed-through instead and would say nothing about
    HER2, so this module is only ever applied to the IHC track.

What this module does NOT do:
    It never assigns a HER2 grade. Grades always come from the dataset's own
    label. Deconvolution is used exclusively to *localise* staining inside an
    image whose grade is already known.
'''

import numpy as np

# Ruifrok & Johnston reference stain vectors (RGB absorbance, unit norm).
# H: haematoxylin, DAB: 3,3'-diaminobenzidine. The residual vector is the cross
# product of the two, which makes the 3x3 system invertible.
HEMATOXYLIN = np.array([0.65, 0.70, 0.29], dtype=np.float64)
DAB = np.array([0.27, 0.57, 0.78], dtype=np.float64)

# Optical density of a fully saturated pixel, used to normalise concentrations
# into a roughly [0, 1] range so thresholds are comparable across scanners.
MAX_OD = 1.8

# Default DAB optical-density cut-offs. These are deliberately conservative and
# are exposed as CLI flags on every script that uses them.
DAB_TISSUE_THRESHOLD = 0.12  # any detectable chromogen
DAB_STRONG_THRESHOLD = 0.45  # dark brown, corresponds to 3+ style staining


def _stain_matrix():
    residual = np.cross(HEMATOXYLIN, DAB)
    residual = residual / np.linalg.norm(residual)
    matrix = np.stack([HEMATOXYLIN, DAB, residual], axis=0)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def rgb_to_od(rgb):
    '''
    rgb: (H, W, 3) uint8
    return: (H, W, 3) float64 optical density
    '''
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f'expected (H, W, 3) RGB, got {rgb.shape}')
    # +1 avoids log(0) for pure black pixels; 256 keeps OD >= 0.
    return -np.log10((rgb.astype(np.float64) + 1.0) / 256.0)


def separate_stains(rgb):
    '''
    rgb: (H, W, 3) uint8
    return: (hematoxylin, dab) concentration maps, each (H, W) float in ~[0, 1]
    '''
    od = rgb_to_od(rgb).reshape(-1, 3)
    concentrations = od @ np.linalg.inv(_stain_matrix())
    concentrations = concentrations.reshape(rgb.shape[0], rgb.shape[1], 3)
    concentrations = np.clip(concentrations / MAX_OD, 0.0, 1.0)
    return concentrations[:, :, 0], concentrations[:, :, 1]


def dab_mask(rgb, threshold=DAB_TISSUE_THRESHOLD):
    '''
    Binary mask of DAB-positive (brown) pixels.

    rgb: (H, W, 3) uint8
    return: (H, W) uint8 mask
    '''
    _, dab = separate_stains(rgb)
    return (dab >= threshold).astype(np.uint8)


def tissue_mask(rgb, threshold=0.10):
    '''
    Binary mask separating tissue from bright glass background, using mean
    optical density. Used to place a whole-tissue box on unstained (HER2 0)
    images, where by definition there is no chromogen to localise.

    rgb: (H, W, 3) uint8
    return: (H, W) uint8 mask
    '''
    return (rgb_to_od(rgb).mean(axis=2) >= threshold).astype(np.uint8)


def dab_fraction(rgb, threshold=DAB_TISSUE_THRESHOLD):
    '''
    Fraction of the image that is DAB-positive. Reported alongside every
    generated sample so a reviewer can audit how much staining a heuristic box
    was actually drawn around.
    '''
    return float(dab_mask(rgb, threshold=threshold).mean())
