import argparse
import os

import pandas as pd
from PIL import Image
from tqdm import tqdm


def save(image_dir, save_csv_path):
    rows = []

    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_files.append(os.path.join(root, file))

    for path in tqdm(list(sorted(image_files))):
        filename = os.path.basename(path)
        try:
            with Image.open(path) as img:
                w, h = img.size
                rows.append({
                    'filename': filename,
                    'width': w,
                    'height': h
                })
        except Exception as e:
            print(f'skip {filename}: {e}')

    df = pd.DataFrame(rows)
    df.to_csv(save_csv_path, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['mimic', 'padchest'])
    args = parser.parse_args()

    if args.dataset == 'mimic':
        image_dir = '../data/mimic-cxr-jpg/2.1.0/files/'
        save_csv_path = '../data/mimic_ori_size.csv'
    elif args.dataset == 'padchest':
        image_dir = '../data/BIMCV-Padchest-GR/Padchest_GR_files/PadChest_GR/'
        save_csv_path = '../data/padchest_512p/ori_size.csv'
    else:
        raise ValueError('Invalid dataset:', args.dataset)

    save(image_dir, save_csv_path)
