import json
import os
import re
import sys
from collections import OrderedDict

import pandas as pd
from ensemble_boxes import weighted_boxes_fusion

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from lofi_utils.info import PADCHEST_GR_LABEL_DICT_inverse


def fuse_boxes(boxes, iou_thr=0.1):
    # https://github.com/philip-mueller/chex/blob/main/src/dataset/vindr_cxr.py
    if not boxes:
        return []
    fused, _, _ = weighted_boxes_fusion(
        [[b] for b in boxes],  # boxes_list
        [[1.0] for _ in boxes],  # scores_list
        [[0] for _ in boxes],  # labels_list
        iou_thr=iou_thr
    )
    return fused.tolist()


def read_chexmask_mimic_cxr(path):
    out_dict = {}
    for _, row in list(pd.read_csv(path).iterrows()):
        out_dict[row['dicom_id']] = row['dice_rca_mean']
    return out_dict


def read_ori_size(ori_size_csv_path):
    ori_size_dict = {}
    for _, row in list(pd.read_csv(ori_size_csv_path).iterrows()):
        ori_size_dict[row['filename']] = (int(row['width']), int(row['height']))
    return ori_size_dict


def read_mimic_img_txt(test_json_path, chexmask_mimic_cxr_path, exclude_normal=False, unique_image=False):
    # read rca
    rca_dict = read_chexmask_mimic_cxr(chexmask_mimic_cxr_path)

    # read json
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    imgs, texts, labels = [], [], []
    image_text_dict_for_unique = OrderedDict()
    valid_samples, low_qualities, excluded_normal, duplicates = 0, 0, 0, 0
    for data in test_data:
        generate_method = data['generate_method']
        id = data['id']
        image_path = data['image'].replace('mimic/', '')  # remove 'mimic/'
        chexpert_labels_dict = data['chexpert_labels']
        view = data['view']
        conversations = data['conversations']
        impression = data['impression']
        if pd.isna(impression):
            impression = ''

        if generate_method != 'gpt4':
            continue

        valid_samples += 1

        # quality check
        dicom_id = os.path.basename(image_path).replace('.jpg', '')
        if (dicom_id not in rca_dict) or (rca_dict[dicom_id] < 0.7):
            low_qualities += 1
            continue
        elif view not in ['PA', 'AP']:
            low_qualities += 1
            continue

        # skip normal
        if exclude_normal and (not any(v == 1 for k, v in chexpert_labels_dict.items() if k != 'No Finding')):
            excluded_normal += 1
            continue

        # check whether the patient and study are identical
        if unique_image and (id in image_text_dict_for_unique):
            duplicates += 1
            prev_dicom_id = os.path.basename(image_text_dict_for_unique[id][0]).replace('.jpg', '')
            if rca_dict[dicom_id] < rca_dict[prev_dicom_id]:
                continue

        findings = conversations[1]['value']
        findings = re.sub(r'\s+', ' ', findings).strip()  # clean text
        impression = re.sub(r'\s+', ' ', impression).strip()  # clean text

        # build report
        report = f'{findings}\n{impression}'
        report = report.strip()  # clean when the impression is empty

        # set dict
        if unique_image:
            image_text_dict_for_unique[id] = (image_path, report, chexpert_labels_dict)  # replace (the report is identical)

        # append list
        imgs.append(image_path)
        texts.append(report)
        labels.append(chexpert_labels_dict)

    print(' * Valid samples:', valid_samples, 'Low qualities:', low_qualities, 'Samples:', len(imgs))

    if unique_image:
        imgs, texts, labels = [], [], []
        for img, txt, l in image_text_dict_for_unique.values():
            imgs.append(img)
            texts.append(txt)
            labels.append(l)
        print(' * Duplicates:', duplicates)

    if exclude_normal:
        print(' * Excluded normal:', excluded_normal)

    return imgs, texts, labels


def read_padchest_gr(json_path, csv_path, ori_size_csv_path=None, png_to_jpg=True, test_split='test'):
    train_data, test_data = [], []

    if test_split == 'val':
        test_split = 'validation'

    # read csv
    train_id_list, test_id_list = [], []
    for _, row in list(pd.read_csv(csv_path).iterrows()):
        ImageID = row['ImageID']
        if png_to_jpg:
            ImageID = ImageID.replace('.png', '.jpg')
        boxes_count = int(row['boxes_count'])
        split = row['split']

        if boxes_count < 1:
            continue

        if split == 'train':
            train_id_list.append(ImageID)
        elif split == test_split:
            test_id_list.append(ImageID)
        else:
            continue

    if ori_size_csv_path is not None:
        ori_size_dict = read_ori_size(ori_size_csv_path)

    # read json
    with open(json_path) as json_data:
        padchest_data = json.load(json_data)

    # build data
    for data in padchest_data:
        ImageID = data['ImageID']
        if png_to_jpg:
            ImageID = ImageID.replace('.png', '.jpg')

        if ImageID in test_id_list:
            is_test = True
        elif ImageID in train_id_list:
            is_test = False
        else:
            continue  # skip (no boxes)

        findings = data['findings']
        if len(findings) == 0:  # for double check
            continue

        for find in findings:
            sentence_en = find['sentence_en']

            if 'boxes' not in find:
                continue

            boxes = find['boxes']

            if len(boxes) == 0:
                continue

            label = PADCHEST_GR_LABEL_DICT_inverse[find['labels'][0]]  # labels should present (use first label only; no multi-label)
            boxes.sort(key=lambda box: box[0])  # sort boxes based on x
            row = {'ImageID': ImageID, 'sentence_en': sentence_en, 'boxes': boxes, 'label': label}

            # add ori_size
            if ori_size_csv_path is not None:
                row['ori_size'] = ori_size_dict[data['ImageID']]  # this should come before png_to_jpg

            if is_test:
                test_data.append(row)
            else:
                train_data.append(row)

    return train_data, test_data
