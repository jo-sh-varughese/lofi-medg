'''
Precompute frozen-encoder vision features so fine-tuning does not recompute
them once per epoch.

The downstream recipe uses --fix_enc and the pipeline applies no image
augmentation, so the encoder output for a given image is identical on every
epoch. Caching it turns N_epochs encoder passes into one. On the 30-epoch
downstream recipe that removes ~97% of the encoder cost, which is the dominant
cost when the encoder is frozen and only LoRA + the projection head train.

This is exact, not an approximation: the cached tensor is precisely what
train_eval.encode_vision would have returned (see --cache_dtype for the one
caveat).

USAGE -- must mirror the training invocation for the flags that affect features
    python tools/precompute_features.py \
        --dataset her2 --her2_dir ./data/her2_512p/ \
        --splits train val \
        --model_dir ./models/ \
        --model_name siglip2-so400m-patch16-512-lofi-medg \
        --resume ./models/lofi-medg/last.pt \
        --pool2x2 \
        --feature_cache_dir ./cache/her2_features/ \
        --batch_size 8

Then add --feature_cache_dir ./cache/her2_features/ to the main.py call.

Safe to interrupt and rerun: already-cached images are skipped.
'''

import argparse
import gc
import os
import sys
from types import SimpleNamespace

import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from lofi_utils.feature_cache import (
    build_manifest, check_manifest, estimate_cache_bytes, read_manifest,
    save_feature, unique_images, write_manifest, cache_file_path,
)
from lofi_utils.model import (
    build_model, checkpoint_lora_scaling, load_checkpoint, load_encoder_state_dict,
    pool2x2 as pool2x2_fn,
)


def build_encoder(args):
    '''
    Mirrors main.py's encoder construction exactly -- same LoRA application,
    same checkpoint load. Any divergence here silently produces features that
    do not match what training would have computed.
    '''
    model, processor = build_model(args.model_name, args.model_dir)

    if processor.image_processor.size['height'] != processor.image_processor.size['width']:
        raise ValueError('Image sizes do not match:', processor.image_processor.size)
    image_size = processor.image_processor.size['height']

    # No apply_lora here, deliberately. A cache is only usable with --fix_enc
    # (main.py enforces this), and --fix_enc zeroes lora_r so training builds a
    # plain frozen encoder and merges the checkpoint's LoRA into its weights.
    # Doing the same here means the cached features are what training would have
    # computed, rather than the arithmetically equivalent but differently
    # rounded output of a LoRA module chain.
    if os.path.isfile(args.resume):
        print(f'Loading encoder weights from {args.resume}')
        ckpt = load_checkpoint(args.resume)
        load_encoder_state_dict(model, ckpt['state_dict'], checkpoint_lora_scaling(ckpt))
        # release the checkpoint before anything else allocates; holding it
        # alongside the model is what gets this OOM-killed on a free runtime
        del ckpt
        gc.collect()
    else:
        # Caching the *base* encoder is legitimate (it matches a run with no
        # --resume), but it is almost never what is wanted, so say so.
        print(f'WARNING: --resume {args.resume!r} is not a file; caching features from the '
              f'base checkpoint. These will NOT match a fine-tuning run that resumes a checkpoint.')

    # Only the vision tower is used here (encode_vision calls model.vision_model).
    # The text tower is a few hundred million parameters of dead weight on a
    # memory-constrained runtime. Dropping it after the state dict is loaded
    # changes nothing numerically.
    if hasattr(model, 'text_model'):
        del model.text_model
        gc.collect()
        print('Released the unused text tower.')

    return model, processor, image_size


def collect_image_paths(args):
    from main import DATASET_MAP

    Dataset = DATASET_MAP.get(args.dataset)
    if Dataset is None:
        raise ValueError('Unknown dataset', args.dataset)

    # The dataset only needs a tokenizer for its text side, which we never
    # touch here; a stub keeps this script from requiring the gated decoder.
    stub_tokenizer = SimpleNamespace(encode=lambda text: [0])

    paths = []
    for split in args.splits:
        dataset = Dataset(
            split=split, processor=None, decoder_tokenizer=stub_tokenizer,
            multimodal_tokens=args.multimodal_tokens, decoder_max_length=args.decoder_max_length,
            args=args, type='qa',
        )
        split_paths = unique_images(dataset)
        print(f'  {split}: {len(dataset)} samples over {len(split_paths)} unique images')
        paths.extend(split_paths)

    seen, out = set(), []
    for path in paths:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


@torch.no_grad()
def encode_batch(model, processor, image_paths, device, dtype, use_pool2x2):
    images = [Image.open(path).convert('RGB') for path in image_paths]
    pixel_values = processor(images=images, return_tensors='pt')['pixel_values']
    pixel_values = pixel_values.to(device=device, dtype=dtype)

    last = model.vision_model(pixel_values=pixel_values).last_hidden_state
    if use_pool2x2:
        # Parameter-free average pool, identical to the first step of
        # ProjectionWrapper.forward. Storing post-pool cuts the cache 4x.
        last = pool2x2_fn(last)
    return last


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f'Device: {device}, compute dtype: {dtype}, cache dtype: {args.cache_dtype}')

    print('Collecting image paths...')
    model, processor, image_size = build_encoder(args)
    model = model.to(device=device, dtype=dtype).eval()

    image_paths = collect_image_paths(args)
    print(f'Total unique images: {len(image_paths)}')

    # One image first, to learn the true feature shape rather than assuming it.
    probe = encode_batch(model, processor, image_paths[:1], device, dtype, args.pool2x2)
    feature_shape = tuple(probe.shape[1:])
    print(f'Feature shape per image: {feature_shape}')

    expected = build_manifest(args.model_name, args.resume, image_size, args.pool2x2,
                              args.cache_dtype, feature_shape)

    found = read_manifest(args.feature_cache_dir)
    if found is not None:
        problems = check_manifest(expected, found)
        if problems:
            raise SystemExit(
                'Existing cache in that directory was built differently:\n  - ' + '\n  - '.join(problems) +
                '\nDelete it or choose another --feature_cache_dir. Refusing to mix.'
            )

    estimated = estimate_cache_bytes(len(image_paths), feature_shape, args.cache_dtype)
    print(f'Estimated cache size: {estimated / (1024 ** 3):.2f} GB')

    todo = [p for p in image_paths if not os.path.isfile(cache_file_path(args.feature_cache_dir, p))]
    print(f'{len(image_paths) - len(todo)} already cached, {len(todo)} to compute')

    numpy_dtype = args.cache_dtype
    for start in tqdm(range(0, len(todo), args.batch_size), desc='Encoding'):
        batch_paths = todo[start:start + args.batch_size]
        features = encode_batch(model, processor, batch_paths, device, dtype, args.pool2x2)
        features = features.to(torch.float32).cpu().numpy().astype(numpy_dtype)
        for path, feature in zip(batch_paths, features):
            save_feature(args.feature_cache_dir, path, feature)

    # Written last: a manifest present means the run got all the way through.
    write_manifest(args.feature_cache_dir, expected)
    print(f'Done. Cache at {args.feature_cache_dir}')
    print(f'Add:  --feature_cache_dir {args.feature_cache_dir}   to your main.py call.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--splits', type=str, nargs='+', default=['train'])
    parser.add_argument('--feature_cache_dir', type=str, required=True)
    parser.add_argument('--cache_dtype', type=str, default='float16', choices=['float16', 'float32'],
                        help='float16 halves disk use; float32 stores exactly what the CPU path computes')
    parser.add_argument('--batch_size', type=int, default=8)

    # must match the training call
    parser.add_argument('--model_dir', type=str, default='./models/')
    parser.add_argument('--model_name', type=str, default='siglip2-so400m-patch16-512')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--pool2x2', action='store_true')
    parser.add_argument('--multimodal_tokens', type=int, default=128)
    parser.add_argument('--decoder_max_length', type=int, default=150)
    parser.add_argument('--lora_target_modules', type=str, nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'out_proj'])
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--head_name', type=str, default='head')

    # dataset dirs
    parser.add_argument('--her2_dir', type=str, default='./data/her2_512p/')
    parser.add_argument('--padchest_image_dir', type=str, default='./data/padchest_512p/')
    parser.add_argument('--tn5000_dir', type=str, default='./data/tn5000_512p/')
    parser.add_argument('--segthor_dir', type=str, default='./data/segthor_512p/')
    parser.add_argument('--medg_dir', type=str, default='./data/medg/')

    run(parser.parse_args())
