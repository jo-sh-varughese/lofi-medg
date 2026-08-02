import argparse
import json
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))

from lofi_utils.data import fuse_boxes
from lofi_utils.misc import convert_pad_space

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='../data/mimic-ext-cxr-qba/1.0.0/exports/A_frontal/qa/')
    parser.add_argument('--file_ext', type=str, default='.qa.json')
    parser.add_argument('--mimic_ori_size_csv_path', type=str, default='../data/mimic/ori_size.csv')
    parser.add_argument('--out_path', type=str, default='../data/mimic/mimic_ext.csv')
    parser.add_argument('--convert_to_pad_bboxes', action='store_true')
    parser.add_argument('--image_size', type=int, default=512)
    args = parser.parse_args()

    if args.convert_to_pad_bboxes:
        args.out_path = args.out_path.replace('.csv', f'_{args.image_size}p.csv')

    # read ori_size csv
    print('Reading ori_size csv...')
    ori_size_dict = {}
    for _, row in list(pd.read_csv(args.mimic_ori_size_csv_path).iterrows()):
        ori_size_dict[row['filename']] = (row['width'], row['height'])

    print('Reading json files:', args.dataset_dir)
    json_path_list = []
    for root, _, fns in os.walk(args.dataset_dir):
        for f in fns:
            if f.endswith(args.file_ext):
                json_path_list.append(os.path.join(root, f))

    rows = {'patient_id': [], 'study_id': [], 'image_id': [], 'text': [], 'width': [], 'height': [], 'bboxes': []}

    for json_path in tqdm(json_path_list):
        unique_image_bboxes_dict = {}

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            print('Skip json:', json_path)
            continue

        # data's key: patient_id, study_id, questions
        patient_id = data['patient_id']
        study_id = data['study_id']
        for question in data['questions']:
            contains_report_answers = question['contains_report_answers']
            if not contains_report_answers:
                continue

            for answer in question['answers']:
                positiveness = answer['positiveness']
                if positiveness != 'pos':
                    continue

                from_report = answer['from_report']
                if not from_report:
                    continue

                answer_text = answer['text']
                localization = answer['localization']
                # certainty = answer['certainty']

                for image_id, loc_data in localization.items():  # in cases where there are multiple frontal X-rays (len(localization)>1)
                    key = f'{patient_id}_{study_id}_{image_id}'

                    bboxes = loc_data['bboxes']
                    if len(bboxes) == 0:
                        print(patient_id, study_id, image_id, answer_text, bboxes)
                        continue

                    # to distinguish multiple frontal X-rays
                    if image_id not in unique_image_bboxes_dict:
                        unique_image_bboxes_dict[image_id] = {}

                    # to skip duplication
                    if answer_text in unique_image_bboxes_dict[image_id]:
                        continue

                    ori_size = ori_size_dict[f'{image_id}.jpg']
                    W, H = ori_size
                    bboxes = [[b[0] / W, b[1] / H, b[2] / W, b[3] / H] for b in bboxes]

                    # to address duplicated bboxes, such as A_frontal/qa/p19/p19934880/s52614570.qa.json
                    bboxes = fuse_boxes(bboxes)
                    if len(bboxes) == 0:
                        continue

                    bboxes.sort(key=lambda box: box[0])  # sort boxes based on x0
                    bboxes = [[round(v, 4) for v in box] for box in bboxes]

                    if args.convert_to_pad_bboxes:
                        bboxes = convert_pad_space(bboxes, ori_size, args.image_size)
                        bboxes = [[int(v * 1000) for v in bbox] for bbox in bboxes]

                    # set bboxes
                    unique_image_bboxes_dict[image_id][answer_text] = str(bboxes)

        for image_id, answer_dict in unique_image_bboxes_dict.items():
            for answer_text, bboxes_str in answer_dict.items():
                width, height = ori_size_dict[f'{image_id}.jpg']
                rows['patient_id'].append(patient_id)
                rows['study_id'].append(study_id)
                rows['image_id'].append(image_id)
                rows['width'].append(width)
                rows['height'].append(height)
                rows['text'].append(answer_text)
                rows['bboxes'].append(bboxes_str)

    # save csv
    pd.DataFrame(rows).to_csv(args.out_path, index=False, encoding='utf-8-sig')
