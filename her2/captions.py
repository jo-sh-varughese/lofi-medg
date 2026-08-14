'''
Region text for the HER2 dataset.

EVERY string in this module is a fixed template. Nothing here is written by a
language model, and nothing here is inferred from pixels. A template is only ever
instantiated with a grade or lesion class that came from the source dataset's own
ground truth, so the clinical claim in the text is as real as the source label --
what is approximate is the box it is attached to, not the statement.

Wording follows the ASCO/CAP HER2 IHC scoring criteria (Wolff et al., 2018
update; 0 vs 1+ distinction per the 2023 update), phrased as a short region
description so it reads naturally in both directions of the LoFi objective:

    grounding : Detect all instances of "<text>".      -> boxes
    captioning: Describe the regions: <boxes>          -> <text>
'''

# Canonical grade keys as they appear in public HER2 datasets.
HER2_GRADES = ['0', '1+', '2+', '3+']

# Grade -> region description. Each phrase names the staining pattern first
# (what is visible in the region) and the score second (what it means).
HER2_REGION_TEXT = {
    '0': 'no membranous staining, HER2 0 negative',
    '1+': 'faint incomplete membranous staining, HER2 1+ negative',
    '2+': 'weak to moderate complete membranous staining, HER2 2+ equivocal',
    '3+': 'strong complete membranous staining, HER2 3+ positive',
}

# Grades whose defining feature is the *absence* of staining. For these there is
# no chromogen to localise, so the region is the tissue as a whole.
UNSTAINED_GRADES = {'0'}

# CAMELYON lesion annotation groups -> region description. These captions are
# generic tumour statements and carry no HER2 claim whatsoever; the boxes under
# them are real pathologist-drawn polygons.
LESION_REGION_TEXT = {
    'tumor': 'metastatic carcinoma in lymph node tissue',
    'metastases': 'metastatic carcinoma in lymph node tissue',
    'normal': 'benign lymph node tissue',
}

# Marks the two tracks in the manifest so evaluation can be broken down by the
# strength of supervision behind each sample.
CONTENT_TYPE_IHC = 'her2-ihc-templated'
CONTENT_TYPE_LESION = 'lesion-polygon-real'


def normalize_grade(grade):
    '''
    Accept the spellings public HER2 datasets actually use ('3+', '3', 3, 'HER2 3+')
    and return a canonical key from HER2_GRADES.
    '''
    text = str(grade).strip().lower().replace('her2', '').strip()
    text = text.replace('score', '').strip()
    if text in ('0', '0+'):
        return '0'
    if text in ('1', '1+'):
        return '1+'
    if text in ('2', '2+'):
        return '2+'
    if text in ('3', '3+'):
        return '3+'
    raise ValueError(f'unrecognised HER2 grade: {grade!r}')


def her2_region_text(grade):
    '''
    grade: one of HER2_GRADES (or anything normalize_grade accepts)
    return: templated region description
    '''
    return HER2_REGION_TEXT[normalize_grade(grade)]


def is_unstained(grade):
    return normalize_grade(grade) in UNSTAINED_GRADES


def lesion_region_text(group_name):
    '''
    group_name: annotation group from a CAMELYON ASAP XML file
    return: templated region description, or None if the group is not one we use
    '''
    return LESION_REGION_TEXT.get(str(group_name).strip().lower())


def all_templates():
    '''
    Flat listing of every template this package can emit. `tools/visualize_her2.py`
    prints it and README_her2.md quotes it, so the full text vocabulary of the
    dataset is auditable without reading the pipeline.
    '''
    templates = [(f'HER2 {grade}', text) for grade, text in HER2_REGION_TEXT.items()]
    templates += [(f'lesion:{group}', text) for group, text in LESION_REGION_TEXT.items()]
    return templates
