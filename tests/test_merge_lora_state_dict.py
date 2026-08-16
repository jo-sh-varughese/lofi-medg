'''
Tests for folding LoRA deltas into base weights at load time.

--fix_enc builds the encoder with plain nn.Linear layers (main.py sets
lora_r = 0), so the released checkpoint's lora_a/lora_b tensors have nowhere to
go. merge_lora_state_dict folds them in instead. The arithmetic must match
tools/merge_lora.py exactly, or the frozen encoder is not the released one.

These need torch, so they skip where it is unavailable.
'''

import pytest

torch = pytest.importorskip('torch')

from lofi_utils.model import checkpoint_lora_scaling, merge_lora_state_dict


def _state_dict(out_dim=4, in_dim=3, rank=2):
    torch.manual_seed(0)
    return {
        'block.q_proj.weight': torch.randn(out_dim, in_dim),
        'block.q_proj.lora_a': torch.randn(rank, in_dim),
        'block.q_proj.lora_b': torch.randn(out_dim, rank),
        'block.norm.weight': torch.randn(out_dim),
    }


class TestMergeArithmetic:
    def test_matches_the_reference_formula(self):
        state = _state_dict()
        scaling = 16 / 16
        expected = state['block.q_proj.weight'] + scaling * (
            state['block.q_proj.lora_b'] @ state['block.q_proj.lora_a']
        )

        merged, count = merge_lora_state_dict(state, scaling)

        assert count == 1
        assert torch.allclose(merged['block.q_proj.weight'], expected, atol=1e-6)

    def test_scaling_is_applied_rather_than_ignored(self):
        state = _state_dict()
        at_one, _ = merge_lora_state_dict(state, 1.0)
        at_four, _ = merge_lora_state_dict(state, 4.0)
        assert not torch.allclose(at_one['block.q_proj.weight'], at_four['block.q_proj.weight'])

    def test_lora_tensors_are_removed(self):
        merged, _ = merge_lora_state_dict(_state_dict(), 1.0)
        assert not [k for k in merged if 'lora' in k]

    def test_unrelated_weights_are_untouched(self):
        state = _state_dict()
        merged, _ = merge_lora_state_dict(state, 1.0)
        assert torch.equal(merged['block.norm.weight'], state['block.norm.weight'])

    def test_input_is_not_mutated(self):
        state = _state_dict()
        original = state['block.q_proj.weight'].clone()
        merge_lora_state_dict(state, 1.0)
        assert 'block.q_proj.lora_a' in state
        assert torch.equal(state['block.q_proj.weight'], original)

    def test_a_state_dict_without_lora_is_a_no_op(self):
        state = {'block.norm.weight': torch.randn(4)}
        merged, count = merge_lora_state_dict(state, 1.0)
        assert count == 0
        assert set(merged) == set(state)

    def test_a_dangling_lora_tensor_raises(self):
        '''Silently skipping would leave the base weight un-updated.'''
        state = _state_dict()
        del state['block.q_proj.lora_b']
        with pytest.raises(KeyError, match='lora_b'):
            merge_lora_state_dict(state, 1.0)


class TestCheckpointScaling:
    def test_reads_alpha_over_r(self):
        assert checkpoint_lora_scaling({'args': {'lora_r': 8, 'lora_alpha': 16}}) == 2.0

    @pytest.mark.parametrize('ckpt', [
        {},
        {'args': {}},
        {'args': {'lora_r': 0, 'lora_alpha': 16}},
    ])
    def test_missing_or_zero_rank_raises_rather_than_guessing(self, ckpt):
        # Defaulting to 1.0 here would apply an unscaled delta and quietly
        # produce an encoder that is not the released one.
        with pytest.raises(ValueError, match='lora_r'):
            checkpoint_lora_scaling(ckpt)
