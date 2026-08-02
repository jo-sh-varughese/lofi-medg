import argparse
import os
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

CPU_COUNT = cpu_count()


def resize(args):
    fp, data_dir, save_dir, size = args
    try:
        rel = os.path.splitext(os.path.relpath(fp, data_dir))[0] + '.jpg'  # always save as .jpg
        out_path = os.path.join(save_dir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # read image
        image = Image.open(fp)
        if image.mode == 'I;16':
            image = Image.fromarray((np.array(image) / 256).astype('uint8'), mode='L')
        image = image.convert('RGB')

        W, H = image.size
        scale = size / max(W, H)  # longer to size
        nw, nh = int(round(W * scale)), int(round(H * scale))
        if (W, H) != (nw, nh):
            image = image.resize((nw, nh), resample=3)  # 3 = Image.LANCZOS

        # save
        image.save(
            out_path,
            format='JPEG',
            quality=100,  # max visual quality
            subsampling=0,  # preserve color detail
            optimize=True,  # compress efficiently
            progressive=True  # better progressive loading
        )
    except Exception as e:
        print(f'Error processing {fp}: {e}')


def resize_with_padding(args):
    fp, data_dir, save_dir, size = args
    try:
        rel = os.path.splitext(os.path.relpath(fp, data_dir))[0] + '.jpg'
        out_path = os.path.join(save_dir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        image = Image.open(fp)
        if image.mode == 'I;16':
            image = Image.fromarray((np.array(image) / 256).astype('uint8'), mode='L')
        image = image.convert('RGB')

        W, H = image.size
        scale = size / max(W, H)  # longer to size
        nw, nh = int(round(W * scale)), int(round(H * scale))
        if (nw, nh) != (W, H):
            image = image.resize((nw, nh), resample=3)  # 3 = Image.LANCZOS

        canvas = Image.new('RGB', (size, size), (0, 0, 0))
        canvas.paste(image, ((size - nw) // 2, (size - nh) // 2))

        # save
        canvas.save(
            out_path,
            'JPEG',
            quality=100,  # max visual quality
            subsampling=0,  # preserve color detail
            optimize=True,  # compress efficiently
            progressive=True  # better progressive loading
        )
    except Exception as e:
        print(f'Error processing {fp}: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', type=str, default='../data/mimic-cxr-jpg/2.1.0/files/')
    parser.add_argument('--save_dir', type=str, default='../data/mimic_512p_good/')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--exts', nargs='+', default=['.jpg', '.jpeg', '.png', '.bmp'])
    parser.add_argument('--num_workers', type=int, default=int(CPU_COUNT * 0.5))
    parser.add_argument('--mimic_csv_filter', type=str, default='../data/mimic/chexmask_mimic_cxr.csv')
    args = parser.parse_args()

    # gather all image files
    files = []
    for root, _, fns in os.walk(args.data_dir):
        for f in fns:
            if os.path.splitext(f)[1].lower() in args.exts:
                files.append(os.path.join(root, f))

    # for mimic
    if ('mimic' in args.save_dir.lower()) and os.path.isfile(args.mimic_csv_filter):
        print('Filtering MIMIC-CXR...')
        rca_dict = {}
        for _, row in list(pd.read_csv(args.mimic_csv_filter).iterrows()):
            rca_dict[row['dicom_id']] = row['dice_rca_mean']

        _files = []
        for fp in tqdm(files):
            dicom_id = os.path.splitext(os.path.basename(fp))[0]
            if (dicom_id not in rca_dict) or (rca_dict[dicom_id] < 0.7):
                continue
            _files.append(fp)
        files = _files

    print(f'Found {len(files)} images to resize.')

    # prepare multiprocessing arguments
    resize_args = [(fp, args.data_dir, args.save_dir, args.size) for fp in files]

    with Pool(args.num_workers) as pool:
        list(tqdm(pool.imap_unordered(resize_with_padding, resize_args), total=len(files), desc=f'Resizing to min side {args.size}'))
