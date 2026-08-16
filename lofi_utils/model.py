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


def build_model(model_name, model_dir):
    ckpt_path = os.path.join(model_dir, model_name)

    # create model and processor
    model = AutoModel.from_pretrained(ckpt_path)
    processor = AutoProcessor.from_pretrained(ckpt_path, use_fast=True)

    return model, processor
