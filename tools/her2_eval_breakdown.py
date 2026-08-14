'''
Break a HER2 detection evaluation down by supervision strength.

`save_detection_eval` stores the per-sample content_type alongside predictions,
but `calc_detection_metrics` reports one pooled number. For this dataset the
pooled number is misleading: it mixes boxes drawn by pathologists (CAMELYON
polygons) with boxes drawn by a threshold (templated IHC regions), and those two
deserve to be read separately.

Usage
    python her2_eval_breakdown.py --pkl_path <RESULT_DIR>/eval_test_her2_ground_ep9.pkl
'''

import argparse
import os
import pickle
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from lofi_utils.metrics import rex_omni_pr_re_f1


def run(args):
    with open(args.pkl_path, 'rb') as f:
        payload = pickle.load(f)

    predictions = payload['predictions']
    targets = payload['targets']
    content_types = payload.get('content_types') or [''] * len(targets)

    groups = {'ALL': list(range(len(targets)))}
    for index, content_type in enumerate(content_types):
        groups.setdefault(content_type or 'unlabelled', []).append(index)

    rows = []
    for name, indices in groups.items():
        scores = rex_omni_pr_re_f1(
            [predictions[i] for i in indices],
            [targets[i] for i in indices],
            iou_thr=args.iou_thr,
        )
        row = {'group': name, 'n': len(indices)}
        row.update({key: round(value * 100, 3) for key, value in scores.items()})
        rows.append(row)

    frame = pd.DataFrame(rows)
    output_path = args.output_path or args.pkl_path.replace('.pkl', '_breakdown.csv')
    frame.to_csv(output_path, index=False)
    print(frame.to_string(index=False))
    print(f'\nWrote {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--iou_thr', type=float, default=0.5)
    args = parser.parse_args()
    run(args)
