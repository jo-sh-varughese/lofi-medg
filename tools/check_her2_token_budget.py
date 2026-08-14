'''
Measure how many decoder tokens a HER2 sample actually costs.

`main.py` sets `decoder_max_length = decoder_max_length + multimodal_tokens`, and
`pad_or_truncate` then cuts anything longer. Truncation removes the *end* of the
target -- the closing code fence and `<end_of_turn>` -- so an over-budget sample
does not raise, it just trains the decoder on a target that never terminates.
Boxes are written as literal text, so cost grows with box count.

This script reproduces the constants in `her2/boxes.py` against the real
tokenizer, over every template in `her2/captions.py`, in both directions of the
LoFi objective, using worst-case four-digit coordinates.

Measured with gemma-3-270m-it:  tokens(n) = 168 + 20n  (plus multimodal_tokens)
    --decoder_max_length 150 (budget 278) -> 5 boxes
    --decoder_max_length 200 (budget 328) -> 8 boxes

Usage
    python check_her2_token_budget.py --tokenizer_path ../models/gemma-3-270m-it/tokenizer.json
'''

import argparse
import os
import sys

from tokenizers import Tokenizer

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from her2 import boxes as her2_boxes
from her2.captions import HER2_REGION_TEXT, LESION_REGION_TEXT

# Duplicated from lofi_utils/gemma.py rather than imported, because that module
# pulls in torch and this check is useful during data preparation. The test suite
# asserts the two stay identical.
CHAT_TEMPLATE = (
    '<start_of_turn>user\n{user}\n<end_of_turn>\n<start_of_turn>model\n{model}\n<end_of_turn>\n'
)


def build_prompts(context, label, boxes):
    '''
    Both directions exactly as lofi_utils/dataset/base.py builds them.
    '''
    question = label[:-1] if label.endswith('.') else label
    grounding = CHAT_TEMPLATE.format(
        user=f'{context}\nDetect all instances of "{question}".', model=f'```\n{boxes}\n```',
    )
    captioning = CHAT_TEMPLATE.format(
        user=f'{context}\nDescribe the regions: {boxes}', model=question,
    )
    return grounding, captioning


def run(args):
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    context = '[multimodal]' * args.multimodal_tokens
    budget = args.decoder_max_length + args.multimodal_tokens

    placeholder = len(tokenizer.encode('[multimodal]').ids)
    expanded = len(tokenizer.encode(context).ids)
    print(f'[multimodal] encodes to {placeholder} id(s); {args.multimodal_tokens}x -> {expanded} tokens')
    if expanded > args.multimodal_tokens + 2:
        print('WARNING: [multimodal] is not a single token in this vocabulary; the context')
        print('         placeholder is consuming more positions than multimodal_tokens.')

    labels = list(HER2_REGION_TEXT.values()) + list(dict.fromkeys(LESION_REGION_TEXT.values()))
    worst_box = [1000, 1000, 1000, 1000]

    print(f'\nbudget = decoder_max_length {args.decoder_max_length} + multimodal_tokens {args.multimodal_tokens} = {budget}\n')
    print(f'{"boxes":>5} {"worst tokens":>13} {"predicted":>10}  fits?')

    largest_fitting = 0
    for count in range(1, args.max_boxes + 1):
        serialized = str([worst_box] * count).replace(' ', '')
        worst = max(
            len(tokenizer.encode(prompt).ids)
            for label in labels
            for prompt in build_prompts(context, label, serialized)
        )
        predicted = her2_boxes.TOKEN_BASE + her2_boxes.TOKEN_PER_BOX * count
        fits = worst <= budget
        if fits:
            largest_fitting = count
        flag = '' if predicted == worst else '   <-- differs from her2/boxes.py constants'
        print(f'{count:>5} {worst:>13} {predicted:>10}  {"yes" if fits else "NO"}{flag}')

    print(f'\nLargest box count that fits intact: {largest_fitting}')
    print(f'her2/boxes.py max_boxes_for() says:  {her2_boxes.max_boxes_for(args.decoder_max_length, args.multimodal_tokens)}')
    print(f'For MAX_BOXES={her2_boxes.MAX_BOXES}, use --decoder_max_length '
          f'{her2_boxes.required_decoder_max_length(her2_boxes.MAX_BOXES, args.multimodal_tokens)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', type=str, default='../models/gemma-3-270m-it/tokenizer.json')
    parser.add_argument('--decoder_max_length', type=int, default=150)
    parser.add_argument('--multimodal_tokens', type=int, default=128)
    parser.add_argument('--max_boxes', type=int, default=10)
    args = parser.parse_args()
    run(args)
