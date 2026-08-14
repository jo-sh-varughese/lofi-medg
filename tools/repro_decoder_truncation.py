'''
Minimal, standalone reproduction of silent target truncation in the LoFi-MedG
downstream fine-tuning recipe.

Run against a fresh clone of lofi-medg:

    python tools/repro_decoder_truncation.py \
        --tokenizer_path ./models/gemma-3-270m-it/tokenizer.json

Depends only on `tokenizers` (already a requirement). It does NOT need torch: if
torch is importable it imports `apply_chat_template` and `pad_or_truncate`
directly from lofi_utils.gemma; otherwise it extracts those two functions
verbatim from lofi_utils/gemma.py with `ast` and execs them. Either way the code
under test is the repo's own, never a reimplementation. The script prints which
path it took.

WHAT IT SHOWS
    README.md, "Downstream Fine-Tuning" uses --decoder_max_length 150 with
    --multimodal_tokens 128. main.py:317 then sets
        decoder_max_length = 150 + 128 = 278
    and BaseDataset.__getitem__ (lofi_utils/dataset/base.py:71-73) tokenizes the
    full chat-templated prompt and calls pad_or_truncate(tokens, 278).

    pad_or_truncate (lofi_utils/gemma.py:520-525) truncates from the RIGHT with
    no warning and no exception. The right end of the target is exactly the part
    that matters: the closing ``` fence and <end_of_turn>.

    Boxes are serialized as literal text (det.py:44 and siblings), so cost grows
    linearly with box count and samples with enough boxes silently lose their
    terminator.
'''

import argparse
import ast
import os
import sys

from tokenizers import Tokenizer

REPO_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
GEMMA_PATH = os.path.join(REPO_ROOT, 'lofi_utils', 'gemma.py')


def load_repo_functions():
    '''
    Return (apply_chat_template, pad_or_truncate, how) taken from the repo.
    '''
    sys.path.insert(0, REPO_ROOT)
    try:
        from lofi_utils.gemma import apply_chat_template, pad_or_truncate
        return apply_chat_template, pad_or_truncate, 'imported directly from lofi_utils.gemma'
    except ImportError as exc:
        with open(GEMMA_PATH, 'rt', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source)
        namespace = {}
        wanted = {'apply_chat_template', 'pad_or_truncate'}
        for node in tree.body:
            # module-level literal assignments the extracted functions may close
            # over (e.g. the truncation-warning dedupe set)
            if isinstance(node, ast.Assign):
                try:
                    exec(ast.unparse(node), namespace)
                except Exception:
                    pass  # anything needing torch etc. is irrelevant here
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                exec(ast.unparse(node), namespace)
        missing = wanted - set(namespace)
        if missing:
            raise RuntimeError(f'not found in {GEMMA_PATH}: {sorted(missing)}')
        return (namespace['apply_chat_template'], namespace['pad_or_truncate'],
                f'extracted verbatim from lofi_utils/gemma.py ({exc.name} unavailable)')


def build_target(apply_chat_template, context, label, n_boxes):
    '''
    Exactly the grounding branch of BaseDataset.__getitem__ (base.py:57-71),
    for a detection dataset. Boxes use worst-case four-digit coordinates.
    '''
    boxes = str([[1000, 1000, 1000, 1000]] * n_boxes).replace(' ', '')
    question = label[:-1] if label.endswith('.') else label
    instruction = f'Detect all instances of "{question}".'
    response = f'```\n{boxes}\n```'
    return apply_chat_template(f'{context}\n{instruction}', response)


def run(args):
    apply_chat_template, pad_or_truncate, how = load_repo_functions()
    print(f'pad_or_truncate / apply_chat_template: {how}\n')

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    encode = lambda text: tokenizer.encode(text).ids
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)

    context = '[multimodal]' * args.multimodal_tokens
    label = 'strong complete membranous staining, HER2 3+ positive'

    print(f'tokenizer vocab size: {tokenizer.get_vocab_size()}')
    print(f'<end_of_turn> id: {encode("<end_of_turn>")}')
    print(f'[multimodal] id:  {encode("[multimodal]")}\n')

    # ---- part 1: measured token cost per box count -----------------------
    print('Measured token cost of the grounding target (worst-case coordinates).')
    print('The intercept depends on the caption text; only the SLOPE is universal.\n')
    print(f'{"boxes":>5} {"tokens":>7} {"per box":>9}')
    previous = None
    for n in (1, 3, 5, 6, 8, 10):
        total = len(encode(build_target(apply_chat_template, context, label, n)))
        rate = '' if previous is None else f'+{(total - previous[1]) // (n - previous[0])}'
        print(f'{n:>5} {total:>7} {rate:>9}')
        previous = (n, total)

    # ---- part 2: the two recipes, same sample ----------------------------
    for max_len in (args.downstream_len, args.pretrain_len):
        budget = max_len + args.multimodal_tokens
        print(f'\n{"=" * 70}')
        print(f'--decoder_max_length {max_len} --multimodal_tokens {args.multimodal_tokens}'
              f'  ->  main.py:317 budget = {budget}')
        print('=' * 70)

        target = build_target(apply_chat_template, context, label, args.n_boxes)
        tokens = encode(target)
        out = pad_or_truncate(tokens, budget, pad_token_id=0)

        print(f'{args.n_boxes} boxes -> {len(tokens)} tokens, budget {budget}, '
              f'{"TRUNCATED" if len(tokens) > budget else "fits"}')
        print(f'\n--- input tail (last 90 chars, [multimodal] context elided) ---')
        print(repr(target[-90:]))
        print(f'\n--- pad_or_truncate output tail, decoded (last 90 chars) ---')
        print(repr(decode(out).replace('[multimodal]', '')[-90:]))

        # The supervised span is the model turn. Check IT terminates, not merely
        # that a fence appears somewhere (the opening fence always survives).
        model_turn = decode(out).split('<start_of_turn>model')[-1]
        model_turn = model_turn.replace('<pad>', '').rstrip()
        print(f'\nmodel turn ends with closing fence : {model_turn.endswith("```") or model_turn.endswith("```<end_of_turn>")}')
        print(f'model turn ends with <end_of_turn> : {model_turn.endswith("<end_of_turn>")}')
        # Upstream pad_or_truncate has no warn path at all; any WARNING printed
        # above means the proposed patch is applied in this checkout.
        print(f'exception raised                   : no (pad_or_truncate returned normally)')

    # ---- part 3: measured boundary, not extrapolated ---------------------
    budget = args.downstream_len + args.multimodal_tokens
    largest = 0
    for n in range(1, 64):
        if len(encode(build_target(apply_chat_template, context, label, n))) <= budget:
            largest = n
        else:
            break
    print(f'\n{"=" * 70}')
    print(f'Measured: largest box count that survives intact at '
          f'--decoder_max_length {args.downstream_len} is {largest}')
    print('=' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', type=str, default='./models/gemma-3-270m-it/tokenizer.json')
    parser.add_argument('--multimodal_tokens', type=int, default=128)
    parser.add_argument('--downstream_len', type=int, default=150, help='README.md downstream recipe')
    parser.add_argument('--pretrain_len', type=int, default=200, help='README.md training recipe')
    parser.add_argument('--n_boxes', type=int, default=8)
    run(parser.parse_args())
