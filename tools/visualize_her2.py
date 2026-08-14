'''
Render converted HER2 samples as a contact sheet for visual sanity checking.

Boxes in the manifest are already in pixel coordinates on the same 512x512
letterboxed canvas as the stored JPEG, so they are drawn directly with no
coordinate transform -- which means this view also verifies that the coordinate
convention is right, not just that the crops look reasonable.

Usage
    python visualize_her2.py --her2_dir ../data/her2_512p/ --split train --num 12
'''

import argparse
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from her2 import captions as her2_captions
from her2 import manifest as her2_manifest

BOX_COLOR = (220, 30, 30)
TEXT_COLOR = (255, 255, 255)
CAPTION_HEIGHT = 52


def _font(size):
    try:  # Pillow >= 10.1 can scale the built-in font
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_sample(dataset_dir, rel_path, pair, thumb_size):
    image = Image.open(os.path.join(dataset_dir, rel_path)).convert('RGB')
    draw = ImageDraw.Draw(image)
    width = max(2, image.size[0] // 160)
    for x0, y0, x1, y1 in pair['box']:
        draw.rectangle([x0, y0, x1, y1], outline=BOX_COLOR, width=width)

    image = image.resize((thumb_size, thumb_size), resample=3)

    canvas = Image.new('RGB', (thumb_size, thumb_size + CAPTION_HEIGHT), (18, 18, 18))
    canvas.paste(image, (0, 0))
    caption = ImageDraw.Draw(canvas)

    wrap_at = max(18, thumb_size // 7)  # the built-in 13px font is ~7px per character
    line, lines = '', []
    for word in pair['label'].split():
        candidate = f'{line} {word}'.strip()
        if len(candidate) > wrap_at:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    if len(lines) > 2:
        lines = lines[:2]
        lines[1] = lines[1][:wrap_at - 1] + '...'

    caption.text((5, thumb_size + 5), '\n'.join(lines), fill=TEXT_COLOR, font=_font(13), spacing=3)
    caption.text(
        (5, thumb_size + 37),
        f'{len(pair["box"])} box(es)   {os.path.basename(rel_path)[:44]}',
        fill=(150, 150, 150), font=_font(11),
    )
    return canvas


def run(args):
    manifest = her2_manifest.read_manifest(args.her2_dir, args.split)
    her2_manifest.validate_manifest(manifest, args.split, args.her2_dir, args.target_size)

    samples = [(rel_path, pair) for rel_path, pairs in manifest['data'].items() for pair in pairs]
    random.Random(args.seed).shuffle(samples)
    samples = samples[:args.num]
    if len(samples) == 0:
        raise ValueError(f'no samples in {args.split}')

    columns = min(args.columns, len(samples))
    rows = (len(samples) + columns - 1) // columns
    sheet = Image.new('RGB', (columns * args.thumb_size, rows * (args.thumb_size + CAPTION_HEIGHT)), (18, 18, 18))

    for index, (rel_path, pair) in enumerate(samples):
        tile = render_sample(args.her2_dir, rel_path, pair, args.thumb_size)
        x = (index % columns) * args.thumb_size
        y = (index // columns) * (args.thumb_size + CAPTION_HEIGHT)
        sheet.paste(tile, (x, y))

    sheet.save(args.output, 'PNG')
    print(f'Wrote {len(samples)} samples -> {args.output}')

    print('\nTemplate vocabulary in use (every caption in this dataset is one of these):')
    for key, text in her2_captions.all_templates():
        print(f'  {key:>16} : {text}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--her2_dir', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--num', type=int)
    parser.add_argument('--columns', type=int)
    parser.add_argument('--thumb_size', type=int)
    parser.add_argument('--target_size', type=int)
    parser.add_argument('--output', type=str)
    parser.add_argument('--seed', type=int)
    parser.set_defaults(
        her2_dir='../data/her2_512p/',
        split='train',
        num=12,
        columns=4,
        thumb_size=256,
        target_size=512,
        output='./her2_samples.png',
        seed=0,
    )
    args = parser.parse_args()
    run(args)
