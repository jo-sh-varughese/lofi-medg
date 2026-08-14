'''
Turn a LoFi run's `train_log.txt` into a CSV, a JSON summary and a loss curve.

main.py appends one line per epoch:
    Train: [0/30] | Total Loss: 1.2345
That is the only training signal the repository records, so this script parses it
rather than adding a second logging path inside the training loop.

Usage
    python plot_her2_training.py --result_dir ../results/her2_.../ [--compare ../results/other_run/]
'''

import argparse
import json
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

LOG_PATTERN = re.compile(r'Train:\s*\[(?P<epoch>\d+)/(?P<epochs>\d+)\]\s*\|\s*Total Loss:\s*(?P<loss>[0-9.eE+-]+)')


def read_log(result_dir):
    log_path = os.path.join(result_dir, 'train_log.txt')
    if not os.path.isfile(log_path):
        raise ValueError(f'no train_log.txt in {result_dir}')

    rows = []
    with open(log_path, 'rt', encoding='utf-8') as f:
        for line in f:
            match = LOG_PATTERN.search(line)
            if match:
                rows.append({
                    'epoch': int(match.group('epoch')),
                    'total_epochs': int(match.group('epochs')),
                    'loss': float(match.group('loss')),
                })
    if len(rows) == 0:
        raise ValueError(f'no parsable epochs in {log_path}')
    return pd.DataFrame(rows).sort_values('epoch').reset_index(drop=True)


def summarize(frame, result_dir):
    best = frame.loc[frame['loss'].idxmin()]
    summary = {
        'result_dir': os.path.abspath(result_dir),
        'epochs_logged': int(len(frame)),
        'first_loss': float(frame['loss'].iloc[0]),
        'last_loss': float(frame['loss'].iloc[-1]),
        'best_loss': float(best['loss']),
        'best_epoch': int(best['epoch']),
    }
    summary['relative_improvement'] = round(
        (summary['first_loss'] - summary['best_loss']) / max(summary['first_loss'], 1e-12), 4
    )

    args_path = os.path.join(result_dir, 'args.json')
    if os.path.isfile(args_path):
        with open(args_path, 'rt', encoding='utf-8') as f:
            run_args = json.load(f)
        summary['config'] = {
            key: run_args.get(key) for key in
            ['dataset', 'model_name', 'epochs', 'batch_size', 'lr', 'lora_r', 'lora_alpha',
             'decoder_lora_r', 'fix_enc', 'finetune_decoder', 'pool2x2', 'multimodal_tokens', 'seed', 'resume']
        }
    return summary


def run(args):
    runs = [args.result_dir] + list(args.compare)

    plt.figure(figsize=(7, 4.5))
    summaries = []
    for result_dir in runs:
        frame = read_log(result_dir)
        frame.to_csv(os.path.join(result_dir, 'train_curve.csv'), index=False)

        summary = summarize(frame, result_dir)
        summaries.append(summary)
        with open(os.path.join(result_dir, 'train_summary.json'), 'wt', encoding='utf-8') as wf:
            json.dump(summary, wf, indent=2)

        plt.plot(frame['epoch'], frame['loss'], marker='o', markersize=3, label=os.path.basename(os.path.normpath(result_dir)))

    plt.xlabel('epoch')
    plt.ylabel('training loss (cross-entropy)')
    plt.title(args.title)
    plt.grid(alpha=0.3)
    if len(runs) > 1:
        plt.legend(fontsize=7)
    plt.tight_layout()

    output_path = args.output or os.path.join(args.result_dir, 'train_curve.png')
    plt.savefig(output_path, dpi=160)
    print(f'Wrote {output_path}')
    for summary in summaries:
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--compare', type=str, nargs='*', default=[])
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--title', type=str, default='HER2 continual fine-tuning')
    args = parser.parse_args()
    run(args)
