import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from lofi_utils.model import LoRALinear, build_model, apply_lora


def merge_lora_linear(model: nn.Module):
    merged_count = 0
    replaced_names = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue

        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model

        merged_weight = module.weight.data + module.scaling * (module.lora_b.data @ module.lora_a.data)
        merged_linear = nn.Linear(
            in_features=module.weight.shape[1],
            out_features=module.weight.shape[0],
            bias=module.bias is not None,
        )
        merged_linear.weight.data.copy_(merged_weight)
        if module.bias is not None:
            merged_linear.bias.data.copy_(module.bias.data)

        setattr(parent, child_name, merged_linear)
        merged_count += 1
        replaced_names.append(name)

    return model, merged_count, replaced_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.set_defaults(
        model_name='siglip2-so400m-patch16-512',
        model_dir='../models/',
        resume='./Results/last.pt',
        output_dir='../models/siglip2-so400m-patch16-512-lofi-medg',
    )
    args = parser.parse_args()

    print('Create model...')
    model, processor = build_model(args.model_name, args.model_dir)

    print('Loading:', args.resume, '...')
    ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
    args_dict = ckpt['args']
    print(json.dumps(args_dict, indent=4))
    if args_dict['lora_r'] > 0:
        model = apply_lora(
            model,
            r=args_dict['lora_r'],
            lora_alpha=args_dict['lora_alpha'],
            target_modules=args_dict['lora_target_modules'],
            head_name=args_dict['head_name']
        )
    model.load_state_dict(ckpt['state_dict'])
    print('LoRA Loaded.')

    print("Merging LoRA weights into base Linear weights...")
    model, merged_count, _ = merge_lora_linear(model)
    print(f"Merged LoRA layers: {merged_count}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Saving merged model to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
