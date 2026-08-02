import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import nibabel.orientations as nio
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from tools.preprocess_utils import seg_slice_to_boxes, check_image_using_pixel, resize_with_padding, save_jpg
from tools.seg_label_info import CT_SEG_FILE_LIST, MRI_SEG_FILE_LIST, NAME_TO_LABEL_DICT
from lofi_utils.misc import convert_pad_space

'''
uv pip install nibabel
'''


def transpose_by_z_index(img, z_index):
    new_axes = [0, 1, 2]
    new_axes.remove(z_index)
    new_axes.append(z_index)
    return img.transpose(new_axes)


def preprocess_mri_subject(subject_dir, save_image_dir, target_size, split, dataset_name):
    save_boxes_dict = {}

    # get subject
    subject = os.path.basename(subject_dir)

    # load nii
    nii = nib.load(os.path.join(subject_dir, 'mri.nii.gz'))
    img_list = np.asarray(nii.dataobj, dtype=np.float32)

    # alternative preprocessing pipeline (no RAS alignment applied)
    img_shape = img_list.shape
    vals, counts = np.unique(img_shape, return_counts=True)
    if np.any(counts == 2):
        z_index = img_shape.index(vals[counts == 1][0])
    else:
        z_index = np.argmin(img_shape)

    img_list = transpose_by_z_index(img_list, z_index)

    seg_list = []
    for seg_filename in MRI_SEG_FILE_LIST:
        single_seg = np.asarray(nib.load(os.path.join(subject_dir, 'segmentations', seg_filename)).dataobj, dtype=np.uint8)
        single_seg = transpose_by_z_index(single_seg, z_index)
        seg_list.append(single_seg)

    # stack seg
    seg = np.stack(seg_list)  # (#label, X, Y, Z)

    # get exist indices
    any_exist_indices = np.where(np.any(seg, axis=(0, 1, 2)))[0].tolist()
    seg_exist_indices = np.where(np.any(seg, axis=(1, 2, 3)))[0].tolist()

    for slice_index in any_exist_indices:
        image = np.flipud(img_list[:, :, slice_index].T)
        if (image.max() - image.min()) == 0:
            continue

        # to uint8 rgb
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)

        # validate images by analyzing pixel values
        if check_image_using_pixel(image):
            continue

        # convert RGB
        image = Image.fromarray(image).convert('RGB')

        # get size
        W, H = image.size

        boxes_dict = {}
        for seg_index in seg_exist_indices:
            seg_slice = np.flipud(seg[seg_index, :, :, slice_index].T)
            boxes = seg_slice_to_boxes(seg_slice)
            if len(boxes) == 0:
                continue

            # get label
            seg_filename = MRI_SEG_FILE_LIST[seg_index]
            label = NAME_TO_LABEL_DICT[seg_filename]

            # convert pad space
            boxes = convert_pad_space(boxes, (W, H), target_size)

            # convert to pixel space
            boxes = [[int(v * target_size) for v in box] for box in boxes]

            # set boxes
            if label not in boxes_dict:
                boxes_dict[label] = []
            boxes_dict[label].extend(boxes)

        if len(boxes_dict) == 0:
            continue

        boxes_list = []
        for label, boxes in boxes_dict.items():
            boxes.sort(key=lambda b: b[0])  # sort boxes
            # it seems better to skip fuse boxes
            boxes_list.append({'box': boxes, 'label': label})

        # set boxes dict
        save_image_filename = f'{subject}_{slice_index:04d}.jpg'
        save_boxes_dict[f'{split}/{dataset_name}/{save_image_filename}'] = boxes_list

        # resize image
        image = resize_with_padding(image, target_size)

        save_jpg(image, os.path.join(save_image_dir, save_image_filename))

    return save_boxes_dict


def preprocess_ct_subject(subject_dir, save_image_dir, target_size, split, dataset_name):
    save_boxes_dict = {}

    # get subject
    subject = os.path.basename(subject_dir)

    # load nii
    nii = nib.load(os.path.join(subject_dir, 'ct.nii.gz'))
    img_list = np.asarray(nii.dataobj, dtype=np.float32)

    # apply orientation
    transform = nio.ornt_transform(nio.io_orientation(nii.affine), nio.axcodes2ornt(('R', 'A', 'S')))
    img_list = nio.apply_orientation(img_list, transform)

    # clip CT HU to [-1000, 1000] since most relevant tissues fall in this range
    # (air -1000, lung -900~-500, fat -100~-50, water 0, soft tissue 30~100, bone 300~1000+)
    img_list = np.clip(img_list, -1000, 1000)

    seg_list = []
    for seg_filename in CT_SEG_FILE_LIST:
        single_seg = np.asarray(nib.load(os.path.join(subject_dir, 'segmentations', seg_filename)).dataobj, dtype=np.uint8)
        single_seg = nio.apply_orientation(single_seg, transform)
        seg_list.append(single_seg)

    # stack seg
    seg = np.stack(seg_list)  # (#label, X, Y, Z)

    # get exist indices
    any_exist_indices = np.where(np.any(seg, axis=(0, 1, 2)))[0].tolist()
    seg_exist_indices = np.where(np.any(seg, axis=(1, 2, 3)))[0].tolist()

    for slice_index in any_exist_indices:
        image = np.flipud(img_list[:, :, slice_index].T)
        if (image.max() - image.min()) == 0:
            continue

        # to uint8 rgb
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)

        # validate images by analyzing pixel values
        if check_image_using_pixel(image):
            continue

        # convert RGB
        image = Image.fromarray(image).convert('RGB')

        # get size
        W, H = image.size

        boxes_dict = {}
        for seg_index in seg_exist_indices:
            seg_slice = np.flipud(seg[seg_index, :, :, slice_index].T)
            boxes = seg_slice_to_boxes(seg_slice)
            if len(boxes) == 0:
                continue

            # get label
            seg_filename = CT_SEG_FILE_LIST[seg_index]
            label = NAME_TO_LABEL_DICT[seg_filename]

            # convert pad space
            boxes = convert_pad_space(boxes, (W, H), target_size)

            # convert to pixel space
            boxes = [[int(v * target_size) for v in box] for box in boxes]

            # set boxes
            if label not in boxes_dict:
                boxes_dict[label] = []
            boxes_dict[label].extend(boxes)

        if len(boxes_dict) == 0:
            continue

        boxes_list = []
        for label, boxes in boxes_dict.items():
            boxes.sort(key=lambda b: b[0])  # sort boxes
            # it seems better to skip fuse boxes
            boxes_list.append({'box': boxes, 'label': label})

        # set boxes dict
        save_image_filename = f'{subject}_{slice_index:04d}.jpg'
        save_boxes_dict[f'{split}/{dataset_name}/{save_image_filename}'] = boxes_list

        # resize image
        image = resize_with_padding(image, target_size)

        save_jpg(image, os.path.join(save_image_dir, save_image_filename))

    return save_boxes_dict


def _run_preprocess(args):
    preprocess_fn, subject_dir, save_dir, target_size, split, dataset_name = args
    return preprocess_fn(
        subject_dir,
        save_dir,
        target_size=target_size,
        split=split,
        dataset_name=dataset_name
    )


def preprocess_dataset(subject_dirs, preprocess_fn, modality, split, medg_dir, dataset_name, target_size, num_workers=None):
    save_dict = {'modality': modality, 'data': {}}

    save_dir = os.path.join(medg_dir, split, dataset_name)
    json_dir = os.path.join(medg_dir, 'json')

    # create directory
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    tasks = [
        (preprocess_fn, subject_dir, save_dir, target_size, split, dataset_name)
        for subject_dir in subject_dirs
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(_run_preprocess, tasks), total=len(tasks)):
            save_dict['data'].update(result)

    json_path = os.path.join(json_dir, f'{split}_{dataset_name}.json')
    with open(json_path, 'wt') as f:
        json.dump(save_dict, f)  # without indent


def read_subject_dir_list(data_dir):
    train_subject_dir_list, test_subject_dir_list = [], []
    csv_path = os.path.join(data_dir, 'meta.csv')
    for _, row in list(pd.read_csv(csv_path, sep=";").iterrows()):
        image_id = row['image_id']
        if ('MRI' in os.path.basename(data_dir)) and (image_id in ['s0332']):
            print('Skip:', image_id)
            continue
        if row['split'] == 'train':
            train_subject_dir_list.append(os.path.join(data_dir, image_id))
        elif row['split'] == 'test':
            test_subject_dir_list.append(os.path.join(data_dir, image_id))
        elif row['split'] == 'val':
            continue
    return train_subject_dir_list, test_subject_dir_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess TotalSegmentator CT/MRI datasets into MedG format.')
    parser.add_argument('--ct_data_dir', type=str, default='../data/Totalsegmentator/Totalsegmentator_dataset_v201/')
    parser.add_argument('--mri_data_dir', type=str, default='../data/Totalsegmentator/TotalsegmentatorMRI_dataset_v200/')
    parser.add_argument('--medg_dir', type=str, default='../data/MedG_512p/')
    parser.add_argument('--target_size', type=int, default=512)
    parser.add_argument('--max_workers', type=int, default=None)
    args = parser.parse_args()

    medg_dir = args.medg_dir

    os.makedirs(medg_dir, exist_ok=True)

    ct_dataset_name = os.path.basename(args.ct_data_dir)
    mri_dataset_name = os.path.basename(args.mri_data_dir)

    ct_train_subject_dir_list, ct_test_subject_dir_list = read_subject_dir_list(args.ct_data_dir)
    mri_train_subject_dir_list, mri_test_subject_dir_list = read_subject_dir_list(args.mri_data_dir)

    max_workers = args.max_workers if args.max_workers is not None else (os.cpu_count() // 4)
    preprocess_dataset(ct_train_subject_dir_list, preprocess_ct_subject, 'CT', 'train', medg_dir, ct_dataset_name, args.target_size, max_workers)
    preprocess_dataset(ct_test_subject_dir_list, preprocess_ct_subject, 'CT', 'test', medg_dir, ct_dataset_name, args.target_size, max_workers)
    preprocess_dataset(mri_train_subject_dir_list, preprocess_mri_subject, 'MRI', 'train', medg_dir, mri_dataset_name, args.target_size, max_workers)
    preprocess_dataset(mri_test_subject_dir_list, preprocess_mri_subject, 'MRI', 'test', medg_dir, mri_dataset_name, args.target_size, max_workers)
