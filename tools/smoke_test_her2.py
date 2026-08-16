'''
End-to-end smoke test for the HER2 pipeline on a laptop.

PURPOSE
    Prove that data -> encoder -> projection -> decoder -> loss -> checkpoint
    runs to completion on real files, with the real checkpoints, before
    spending scarce GPU time on it. It runs a handful of samples for one epoch.

    This produces NO RESULT. The loss it prints is meaningless at this scale and
    must never appear in RESULTS_her2.md. The only question it answers is
    "does the wiring work", which is currently untested end to end.

REQUIREMENTS
    - the HER2 dataset built by tools/preprocess_bci.py
    - ./models/<encoder> and ./models/gemma-3-270m-it (the decoder is gated;
      accept the licence on HuggingFace first)
    - torch. No GPU needed -- a few samples on CPU is the point.

USAGE
    python tools/smoke_test_her2.py --her2_dir ./data/her2_512p/ --model_dir ./models/

Add --with_cache to also exercise the precompute path (tools/precompute_features.py
followed by a cached training step), which is what a Colab run will use.
'''

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')


def run_step(name, command):
    print(f'\n{"=" * 70}\n{name}\n{"=" * 70}')
    print(' '.join(command) + '\n')
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f'\nFAILED: {name} (exit {result.returncode})')
        return False
    print(f'\nOK: {name}')
    return True


def checkpoint_lora_flags(resume):
    '''
    Build the --lora_* flags that match the checkpoint.

    main.py applies LoRA from its CLI args *before* it resumes, so a mismatch
    with the checkpoint fails at load_state_dict. The released LoFi-MedG
    checkpoint targets q_proj k_proj v_proj out_proj fc1 fc2; the CLI default is
    only the first four. Reading the checkpoint keeps the two in step without
    anyone having to remember the longer list.
    '''
    if not os.path.isfile(resume):
        return []

    sys.path.append(REPO_ROOT)
    from lofi_utils.model import read_checkpoint_args

    args_dict = read_checkpoint_args(resume)
    if not args_dict:
        print('Checkpoint carries no args dict; leaving the LoRA flags at their defaults.')
        return []

    flags = []
    if 'lora_target_modules' in args_dict:
        flags += ['--lora_target_modules'] + list(args_dict['lora_target_modules'])
    for key in ('lora_r', 'lora_alpha', 'head_name'):
        if key in args_dict:
            flags += [f'--{key}', str(args_dict[key])]
    print(f'LoRA flags taken from the checkpoint: {" ".join(flags)}')
    return flags


def main(args):
    python = sys.executable
    result_dir = os.path.join(args.result_dir, 'smoke')
    os.makedirs(os.path.join(REPO_ROOT, result_dir), exist_ok=True)

    common = [
        '--dataset', 'her2',
        '--her2_dir', args.her2_dir,
        '--model_dir', args.model_dir,
        '--model_name', args.model_name,
        '--multimodal_tokens', '128',
        '--decoder_max_length', '200',
        '--pool2x2',
    ]
    if args.resume:
        common += ['--resume', args.resume]
        common += checkpoint_lora_flags(args.resume)

    steps = []

    if args.with_cache:
        steps.append((
            'Precompute frozen-encoder features (train split, few images)',
            [python, 'tools/precompute_features.py', '--splits', 'train',
             '--feature_cache_dir', args.feature_cache_dir, '--batch_size', '2'] + common,
        ))

    train_command = [
        python, 'main.py',
        '--epochs', '1',
        '--batch_size', str(args.batch_size),
        '--num_workers', '0',          # workers add nothing at this size and hide tracebacks
        '--limit_samples', str(args.limit_samples),
        '--lr', '3e-4',
        '--cos_eta_min', '0.1',
        '--finetune_decoder',
        '--fix_enc',
        '--seed', '42',
        '--result_dir', result_dir,
    ] + common
    if args.with_cache:
        train_command += ['--feature_cache_dir', args.feature_cache_dir]
    steps.append(('Train one epoch on a few samples', train_command))

    for name, command in steps:
        if not run_step(name, command):
            print('\nSmoke test FAILED. Fix this before using any GPU time.')
            return 1

    print(f'\n{"=" * 70}')
    print('Smoke test passed: the pipeline runs end to end on real data.')
    print('This is NOT a result. Do not record the loss anywhere.')
    print('=' * 70)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--her2_dir', type=str, default='./data/her2_512p/')
    parser.add_argument('--model_dir', type=str, default='./models/')
    parser.add_argument('--model_name', type=str, default='siglip2-so400m-patch16-512-lofi-medg')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--result_dir', type=str, default='./results/')
    parser.add_argument('--feature_cache_dir', type=str, default='./cache/her2_smoke/')
    parser.add_argument('--limit_samples', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--with_cache', action='store_true',
                        help='also exercise the precompute + cached-training path')
    sys.exit(main(parser.parse_args()))
