import gc
import os

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from transformers.models.siglip2.modeling_siglip2 import Siglip2MultiheadAttentionPoolingHead


def pool2x2(last):
    B, N, C = last.shape
    s = int(N ** 0.5)
    return F.avg_pool2d(last.view(B, s, s, C).permute(0, 3, 1, 2), 2, 2).permute(0, 2, 3, 1).reshape(B, -1, C)


class ProjectionWrapper(nn.Module):
    def __init__(self, vision_model_config, multimodal_tokens, emb_dim, use_pool2x2):
        super().__init__()
        self.poolings = nn.ModuleList([Siglip2MultiheadAttentionPoolingHead(vision_model_config) for _ in range(multimodal_tokens)])  # better than multi-probe attention pooling
        self.projection = nn.Sequential(nn.Linear(vision_model_config.hidden_size, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim))
        self.use_pool2x2 = use_pool2x2

    def forward(self, last, attention_mask=None, already_pooled=False):
        # (for efficiency) following the 4x4 average pooling used in Gemma 3 for 896-resolution inputs
        # already_pooled: features came from the precomputed cache, which stores
        # them post-pool (see lofi_utils/feature_cache.py); pooling twice would
        # silently halve the spatial grid again.
        if self.use_pool2x2 and not already_pooled:
            last = pool2x2(last)  # 1024 -> 256

        # attention pooling
        multimodal_tokens = torch.stack([p(last, attention_mask=attention_mask) for p in self.poolings], dim=1)

        # projection
        multimodal_tokens = self.projection(multimodal_tokens)

        return multimodal_tokens


####################
# LoRA
####################
class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16.0, use_bias=False):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.lora_a = nn.Parameter(torch.empty(rank, in_dim))
        self.lora_b = nn.Parameter(torch.empty(out_dim, rank))
        self.bias = nn.Parameter(torch.empty(out_dim)) if use_bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        lora_result = F.linear(F.linear(x, self.lora_a), self.lora_b)
        return result + self.scaling * lora_result


def apply_lora(model, r, lora_alpha, target_modules, head_name, debug=False):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules) and (head_name not in name):
            if debug:
                print('LoRA:', name)
            parent = model.get_submodule('.'.join(name.split('.')[:-1]))
            child_name = name.split('.')[-1]
            lora_layer = LoRALinear(
                in_dim=module.in_features,
                out_dim=module.out_features,
                rank=r,
                alpha=lora_alpha,
                use_bias=module.bias is not None,
            )
            lora_layer.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_layer.bias.data = module.bias.data.clone()
            setattr(parent, child_name, lora_layer)
        else:
            if debug and hasattr(module, 'weight'):
                print(' - No LoRA:', name)

    # freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze LoRA parameters and the classifier
    for name, param in model.named_parameters():
        if ('lora_a' in name) or ('lora_b' in name) or (head_name in name):
            param.requires_grad = True
            if debug:
                print('* Unfrozen param:', name)

    if debug:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f'Trainable params: {trainable:,} | Total params: {total:,} | Trainable %: {100 * trainable / total:.4f}')

    return model


def load_checkpoint(ckpt_path):
    '''
    Load a checkpoint without materialising a second full copy of the weights.

    torch.load normally reads every tensor into RAM, so `model` plus `ckpt` both
    resident peaks at roughly twice the model size -- about 9 GB for
    siglip2-so400m in fp32, which the OOM killer ends on a 12.7 GB Colab
    runtime (exit -9). mmap=True keeps tensors backed by the file until each one
    is copied into place.

    Falls back to a plain load for checkpoints written before torch's zipfile
    serialisation, which mmap requires.
    '''
    try:
        return torch.load(ckpt_path, map_location='cpu', weights_only=False, mmap=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        print(f'mmap load unavailable ({exc}); falling back to a full read')
        return torch.load(ckpt_path, map_location='cpu', weights_only=False)


def read_checkpoint_args(ckpt_path):
    '''
    Return the args dict a checkpoint was trained with, or None if absent.

    The LoRA topology must match the checkpoint exactly or load_state_dict
    raises: the released LoFi-MedG checkpoint targets q_proj k_proj v_proj
    out_proj fc1 fc2, while main.py's --lora_target_modules default is only the
    first four. Trusting the checkpoint over the CLI default is what
    tools/merge_lora.py and main.py's evaluation path already do.
    '''
    ckpt = load_checkpoint(ckpt_path)
    args_dict = ckpt.get('args')
    del ckpt
    gc.collect()
    return args_dict


def checkpoint_lora_scaling(ckpt):
    '''
    The LoRA scaling (alpha / r) the checkpoint's encoder was trained with.

    Read from the checkpoint because callers that freeze the encoder have
    already zeroed their own lora_r/lora_alpha and can no longer supply it.
    '''
    args_dict = ckpt.get('args') or {}
    r, alpha = args_dict.get('lora_r'), args_dict.get('lora_alpha')
    if not r:
        raise ValueError(
            'Checkpoint records no lora_r, so its LoRA deltas cannot be scaled correctly. '
            'Merge them with tools/merge_lora.py first and load the merged encoder.'
        )
    return alpha / r


def lora_delta(lora_a, lora_b, scaling):
    '''
    scaling * (B @ A), in float32.

    The float32 cast is deliberate: deltas stored in a reduced precision would
    otherwise lose most of the update. Shared by both merge paths so the
    arithmetic cannot drift between them.
    '''
    return scaling * (lora_b.float() @ lora_a.float())


def lora_prefixes(state_dict):
    '''The module prefixes in a state dict that carry LoRA tensors.'''
    return [k[: -len('.lora_a')] for k in state_dict if k.endswith('.lora_a')]


def merge_lora_state_dict(state_dict, scaling):
    '''
    Fold LoRA deltas into the base weights of a state dict, in place of the
    module surgery tools/merge_lora.py performs on a live model.

    Needed because --fix_enc sets lora_r = 0 (main.py, line 460), so the encoder
    is built with plain nn.Linear layers and has nowhere to put the checkpoint's
    lora_a/lora_b tensors. Folding them in gives that same frozen encoder the
    fine-tuned weights it is supposed to have, and keeps the checkpoint's
    projection and decoder -- which merging to a new model directory would
    discard.

    Same arithmetic as merge_lora_linear: W' = W + scaling * (B @ A).

    This is the reference form, and what the tests pin. load_encoder_state_dict
    does not call it: on a memory-constrained runtime the merged copy it returns
    is itself the problem, so that path merges into the model's parameters
    instead. Use this one where a plain merged state dict is what is wanted.

    Returns (merged_state_dict, merged_count).
    '''
    merged = dict(state_dict)
    count = 0

    for key in [k for k in merged if k.endswith('.lora_a')]:
        prefix = key[: -len('.lora_a')]
        b_key, weight_key = f'{prefix}.lora_b', f'{prefix}.weight'
        if b_key not in merged or weight_key not in merged:
            raise KeyError(f'{prefix}: lora_a present but {b_key!r} or {weight_key!r} is missing')

        weight = merged[weight_key]
        delta = lora_delta(merged[key], merged[b_key], scaling)
        merged[weight_key] = (weight.float() + delta).to(weight.dtype)

        del merged[key], merged[b_key]
        count += 1

    return merged, count


def load_encoder_state_dict(model, state_dict, scaling):
    '''
    Load encoder weights, folding in LoRA if the model was built without it.

    A model built with apply_lora takes the checkpoint as-is; one built under
    --fix_enc needs the deltas merged first. Deciding from the model rather than
    from a flag keeps the two callers (main.py and tools/precompute_features.py)
    from drifting apart.

    Merges into the model's own parameters rather than into a copy of the state
    dict. Building a merged dict means ~2 GB of extra float32 weights alive
    beside the 4.4 GB model, which is enough to get the training run OOM-killed
    on a 12.7 GB runtime.
    '''
    has_lora_modules = any(isinstance(m, LoRALinear) for m in model.modules())
    prefixes = lora_prefixes(state_dict)

    if not prefixes or has_lora_modules:
        model.load_state_dict(state_dict)
        return model

    # Load the base weights first, holding back the LoRA tensors the plain
    # nn.Linear layers have no home for. strict=False would also hide a genuinely
    # missing weight, so the result is checked rather than trusted.
    lora_keys = {f'{p}.{suffix}' for p in prefixes for suffix in ('lora_a', 'lora_b')}
    base = {k: v for k, v in state_dict.items() if k not in lora_keys}
    missing, unexpected = model.load_state_dict(base, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f'Encoder weights do not match the model after setting LoRA aside. '
            f'Missing: {list(missing)[:5]}. Unexpected: {list(unexpected)[:5]}.'
        )

    params = dict(model.named_parameters())
    with torch.no_grad():
        for prefix in prefixes:
            weight_key = f'{prefix}.weight'
            if weight_key not in params:
                raise KeyError(f'{prefix}: checkpoint has LoRA for a weight the model does not have')
            weight = params[weight_key]
            delta = lora_delta(state_dict[f'{prefix}.lora_a'], state_dict[f'{prefix}.lora_b'], scaling)
            weight.add_(delta.to(weight.dtype))
            del delta

    print(f'Merged {len(prefixes)} LoRA layers into the base weights (frozen encoder, no LoRA modules).')
    return model


def build_model(model_name, model_dir):
    ckpt_path = os.path.join(model_dir, model_name)

    # create model and processor
    model = AutoModel.from_pretrained(ckpt_path)
    processor = AutoProcessor.from_pretrained(ckpt_path, use_fast=True)

    return model, processor
