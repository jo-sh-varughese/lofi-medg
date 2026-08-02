import argparse
import ast
import json
import os
import sys
from collections import OrderedDict
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image
from scipy import sparse

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from tools.preprocess_utils import seg_slice_to_boxes, check_image_using_pixel, resize_with_padding, save_jpg
from tools.seg_label_info import NAME_TO_LABEL_DICT, MODALITY_NAME_TO_LABEL_DICT
from lofi_utils.misc import convert_pad_space


def preprocess_samples(data_list, labels_dict, dataset_dir, save_image_dir, target_size, split):
    save_boxes_dict = OrderedDict()

    # get dataset name
    dataset_name = os.path.basename(dataset_dir)

    for data in data_list:
        image_filename = data['image']
        image_path = os.path.join(dataset_dir, image_filename)
        label_path = os.path.join(dataset_dir, data['label'])

        save_image_filename = image_filename.replace('.png', '.jpg').replace('image/', '').replace('/', '_')  # image/x/img.jpg -> x_img.jpg

        try:
            # read image
            image = Image.open(image_path)

            # validate images by analyzing pixel values
            if check_image_using_pixel(np.asarray(image.convert('L'))):
                raise ValueError()
        except:
            continue

        try:
            # read segmentation
            gt_shape = ast.literal_eval(os.path.basename(label_path).split('.')[-2])
            allmatrix_sp = sparse.load_npz(label_path)
            seg = allmatrix_sp.toarray().reshape(gt_shape)
            seg = seg[..., 0]  # (..., 1) -> (...)

            # get shape
            W, H = image.size
            assert (H, W) == seg.shape[1:3]
            assert seg.shape[0] == len(labels_dict)

            seg_exist_indices = np.where(np.any(seg, axis=(1, 2)))[0].tolist()

            boxes_dict = {}
            for seg_index in seg_exist_indices:
                seg_slice = seg[seg_index]
                boxes = seg_slice_to_boxes(seg_slice)
                if len(boxes) == 0:
                    continue

                # get label_name
                label_name = labels_dict[str(seg_index + 1)]  # the segmentation does not include the background

                # get label
                label = NAME_TO_LABEL_DICT[label_name]

                # convert pad space
                boxes = convert_pad_space(boxes, (W, H), target_size)

                # convert to pixel space
                boxes = [[int(v * target_size) for v in box] for box in boxes]

                # set boxes
                if label not in boxes_dict:
                    boxes_dict[label] = []
                boxes_dict[label].extend(boxes)

            # check invalid seg
            if len(boxes_dict) == 0:
                raise ValueError()
        except:
            continue

        boxes_list = []
        for label, boxes in boxes_dict.items():
            boxes.sort(key=lambda b: b[0])  # sort boxes
            # it seems better to skip fuse boxes
            boxes_list.append({'box': boxes, 'label': label})

        # set boxes dict
        save_boxes_dict[f'{split}/{dataset_name}/{save_image_filename}'] = boxes_list

        # resize image
        image = resize_with_padding(image, target_size)

        save_jpg(image, os.path.join(save_image_dir, save_image_filename))

    return save_boxes_dict


def preprocess_dataset(dataset_dir, medg_dir, target_size):
    # get dataset name
    dataset_name = os.path.basename(dataset_dir)

    print(f'Preprocess {dataset_name}...')

    # set directory
    save_train_image_dir = os.path.join(medg_dir, 'train', dataset_name)
    save_test_image_dir = os.path.join(medg_dir, 'test', dataset_name)
    medg_json_dir = os.path.join(medg_dir, 'json')

    # create directory
    os.makedirs(save_train_image_dir, exist_ok=True)
    os.makedirs(save_test_image_dir, exist_ok=True)
    os.makedirs(medg_json_dir, exist_ok=True)

    # read dataset json
    dataset_json_path = os.path.join(dataset_dir, 'dataset.json')
    with open(dataset_json_path, 'r') as f:
        dataset_json = json.load(f)

    # get samples
    training_list = dataset_json['training']
    test_list = dataset_json['test']

    # get meta data
    labels_dict = dataset_json['labels']
    modality = dataset_json['modality']

    del labels_dict['0']  # exclude background (left to raise a KeyError)

    save_train_dict, save_test_dict = {}, {}
    save_train_dict['modality'] = MODALITY_NAME_TO_LABEL_DICT[modality['0']]
    save_test_dict['modality'] = MODALITY_NAME_TO_LABEL_DICT[modality['0']]

    # preprocess
    save_train_dict['data'] = preprocess_samples(training_list, labels_dict, dataset_dir, save_image_dir=save_train_image_dir, target_size=target_size, split='train')
    save_test_dict['data'] = preprocess_samples(test_list, labels_dict, dataset_dir, save_image_dir=save_test_image_dir, target_size=target_size, split='test')

    # save dataset json
    with open(os.path.join(medg_json_dir, f'train_{dataset_name}.json'), 'wt') as f:
        json.dump(save_train_dict, f, indent=4)
    with open(os.path.join(medg_json_dir, f'test_{dataset_name}.json'), 'wt') as f:
        json.dump(save_test_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imed_dir', type=str, default='../data/IMed-361M/')
    parser.add_argument('--medg_dir', type=str, default='../data/MedG_512p/')
    parser.add_argument('--target_size', type=int, default=512)
    args = parser.parse_args()

    medg_dir = args.medg_dir

    os.makedirs(medg_dir, exist_ok=False)

    dataset_dir_list = []
    for dataset_name in sorted(os.listdir(args.imed_dir)):
        dataset_dir = os.path.join(args.imed_dir, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        dataset_dir_list.append(dataset_dir)

    args_list = [(dataset_dir, medg_dir, args.target_size) for dataset_dir in dataset_dir_list]
    num_workers = min(cpu_count() // 2, len(args_list))

    with Pool(processes=num_workers) as pool:
        pool.starmap(preprocess_dataset, args_list)
