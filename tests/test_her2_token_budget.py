'''
Guards on the decoder token budget.

Over-budget samples do not raise: `pad_or_truncate` silently cuts the tail off the
target, removing the closing fence and `<end_of_turn>`. These tests pin the
measured cost model and the box-count/decoder_max_length pairing that follows from
it, so a later change to MAX_BOXES or to the caption templates cannot quietly
reintroduce truncation.

The constants were measured against gemma-3-270m-it with
tools/check_her2_token_budget.py, which reproduces them exactly at every box count.
'''

import os
import re

from her2 import boxes as her2_boxes
from tools.check_her2_token_budget import CHAT_TEMPLATE, build_prompts


def test_cost_model_matches_the_measured_values():
    # measured: 188 tokens for 1 box, 328 for 8, over every template, worst-case digits
    assert her2_boxes.TOKEN_BASE + her2_boxes.TOKEN_PER_BOX * 1 == 188
    assert her2_boxes.TOKEN_BASE + her2_boxes.TOKEN_PER_BOX * 8 == 328


def test_default_recipe_budget_fits_five_boxes():
    assert her2_boxes.max_boxes_for(decoder_max_length=150, multimodal_tokens=128) == 5
    assert her2_boxes.max_boxes_for(decoder_max_length=200, multimodal_tokens=128) == 8


def test_max_boxes_is_paired_with_a_documented_decoder_max_length():
    needed = her2_boxes.required_decoder_max_length(her2_boxes.MAX_BOXES, multimodal_tokens=128)
    assert needed == 200, 'MAX_BOXES changed: update the --decoder_max_length in README_her2.md'
    assert her2_boxes.max_boxes_for(needed, 128) >= her2_boxes.MAX_BOXES


def test_the_two_helpers_are_consistent():
    for max_boxes in range(1, 13):
        needed = her2_boxes.required_decoder_max_length(max_boxes, 128)
        assert her2_boxes.max_boxes_for(needed, 128) >= max_boxes
        if needed > 0:
            assert her2_boxes.max_boxes_for(needed - her2_boxes.TOKEN_PER_BOX, 128) < max_boxes


def test_budget_tool_builds_the_same_prompts_as_the_loader():
    '''
    tools/check_her2_token_budget.py copies the chat template rather than importing
    it (that module pulls in torch). Assert the copy still matches the original.
    '''
    gemma_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'lofi_utils', 'gemma.py')
    with open(gemma_path, 'rt', encoding='utf-8') as f:
        source = f.read()

    match = re.search(r'def apply_chat_template.*?(?=\n\n|\nclass |\ndef )', source, re.S)
    assert match, 'apply_chat_template not found in lofi_utils/gemma.py'
    namespace = {}
    exec(match.group(0), namespace)  # pure string function, no torch involved

    original = namespace['apply_chat_template']('USER TEXT', 'MODEL TEXT')
    copied = CHAT_TEMPLATE.format(user='USER TEXT', model='MODEL TEXT')
    assert copied == original


def test_prompts_cover_both_directions_of_the_objective():
    grounding, captioning = build_prompts('[multimodal]', 'strong complete membranous staining, HER2 3+ positive', '[[1,2,3,4]]')

    assert 'Detect all instances of' in grounding and '[[1,2,3,4]]' in grounding
    assert 'Describe the regions:' in captioning and 'HER2 3+ positive' in captioning


def test_trailing_period_is_stripped_exactly_once_like_the_loader():
    grounding, _ = build_prompts('[multimodal]', 'a finding.', '[[1,2,3,4]]')
    assert 'Detect all instances of "a finding".' in grounding
