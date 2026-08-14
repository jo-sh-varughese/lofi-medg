'''
Collect slide-level HER2 labels from HEROHE and TCGA-BRCA.

These labels are deliberately kept OUT of the grounding manifests. Both cohorts
are H&E; their HER2 status was determined by a separate IHC/FISH assay performed
on glass that is not in the image archive. Attaching a HER2 staining caption to a
region of an H&E slide would assert a finding the pixels cannot support, so this
script emits a flat CSV for a slide-level classification probe on the encoder's
features instead -- a different task, evaluated separately.

Output: her2_slide_labels.csv with columns
    slide_id, cohort, her2_status, her2_ihc_score, stain, source_file

Usage
    python preprocess_her2_slide_labels.py \
        --herohe_csv ../data/HEROHE/HEROHE_TrainGroundTruth.csv \
        --tcga_clinical ../data/TCGA-BRCA/clinical.tsv \
        --output_path ../data/her2/her2_slide_labels.csv
'''

import argparse
import os

import pandas as pd

# Column names vary between GDC clinical exports and cBioPortal downloads.
TCGA_ID_COLUMNS = ['case_submitter_id', 'submitter_id', 'bcr_patient_barcode', 'Patient Identifier']
TCGA_STATUS_COLUMNS = ['her2_status_by_ihc', 'HER2 ihc status', 'IHC-HER2']
TCGA_SCORE_COLUMNS = ['her2_ihc_score', 'HER2 ihc score']

NULL_VALUES = {'', 'na', 'n/a', 'nan', 'not reported', 'unknown', '--', '[not available]', '[unknown]', 'indeterminate'}


def _first_column(frame, candidates):
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def _clean(value):
    text = str(value).strip()
    return '' if text.lower() in NULL_VALUES else text


def read_herohe(path):
    '''
    HEROHE ground truth: one row per case with a binary HER2 status
    (0 = 0/1+ negative, 1 = 2+/3+ positive). The challenge never released a
    four-tier score, so her2_ihc_score is left empty.
    '''
    frame = pd.read_csv(path)
    id_column = _first_column(frame, ['caseID', 'CaseID', 'case_id', 'Slide', 'slide_id'])
    status_column = _first_column(frame, ['HER2status', 'HER2', 'status', 'label'])
    if id_column is None or status_column is None:
        raise ValueError(f'unexpected HEROHE columns: {list(frame.columns)}')

    rows = []
    for _, row in frame.iterrows():
        status = _clean(row[status_column]).lower()
        if status in ('1', '1.0', 'positive', 'pos'):
            status = 'Positive'
        elif status in ('0', '0.0', 'negative', 'neg'):
            status = 'Negative'
        else:
            continue
        rows.append({
            'slide_id': _clean(row[id_column]),
            'cohort': 'HEROHE',
            'her2_status': status,
            'her2_ihc_score': '',
            'stain': 'H&E',
            'source_file': os.path.basename(path),
        })
    return rows


def read_tcga(path):
    '''
    TCGA-BRCA clinical export. `her2_status_by_ihc` is present for most cases;
    `her2_ihc_score` (0/1+/2+/3+) is present for a subset.
    '''
    separator = '\t' if path.lower().endswith(('.tsv', '.txt')) else ','
    frame = pd.read_csv(path, sep=separator, dtype=str, comment=None)
    id_column = _first_column(frame, TCGA_ID_COLUMNS)
    status_column = _first_column(frame, TCGA_STATUS_COLUMNS)
    score_column = _first_column(frame, TCGA_SCORE_COLUMNS)
    if id_column is None or status_column is None:
        raise ValueError(f'unexpected TCGA clinical columns: {list(frame.columns)[:20]}')

    rows, seen = [], set()
    for _, row in frame.iterrows():
        case_id = _clean(row[id_column])
        status = _clean(row[status_column])
        if not case_id or not status or case_id in seen:
            continue
        seen.add(case_id)
        rows.append({
            'slide_id': case_id,
            'cohort': 'TCGA-BRCA',
            'her2_status': status.capitalize(),
            'her2_ihc_score': _clean(row[score_column]) if score_column else '',
            'stain': 'H&E',
            'source_file': os.path.basename(path),
        })
    return rows


def run(args):
    rows = []
    if args.herohe_csv and os.path.isfile(args.herohe_csv):
        rows.extend(read_herohe(args.herohe_csv))
    if args.tcga_clinical and os.path.isfile(args.tcga_clinical):
        rows.extend(read_tcga(args.tcga_clinical))

    if len(rows) == 0:
        raise ValueError('no slide-level labels read; check --herohe_csv and --tcga_clinical')

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_path, index=False)

    print(f'Wrote {len(frame)} slide-level labels -> {args.output_path}')
    print(frame.groupby(['cohort', 'her2_status']).size().to_string())
    print('\nNOTE: these labels are for a slide-level classification probe only.')
    print('They are H&E cohorts and are never used as HER2 grounding captions.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--herohe_csv', type=str)
    parser.add_argument('--tcga_clinical', type=str)
    parser.add_argument('--output_path', type=str)
    parser.set_defaults(
        herohe_csv='../data/HEROHE/HEROHE_TrainGroundTruth.csv',
        tcga_clinical='../data/TCGA-BRCA/clinical.tsv',
        output_path='../data/her2/her2_slide_labels.csv',
    )
    args = parser.parse_args()
    run(args)
