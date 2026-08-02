import ast
import json
import os
import sys
from collections import OrderedDict

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

from lofi_utils.dataset.base import BaseDataset
from lofi_utils.data import read_mimic_img_txt, read_padchest_gr
from lofi_utils.misc import convert_pad_space, get_target_annotated_size


class MedGDataset(BaseDataset):
    def __init__(self, split, processor, decoder_tokenizer, multimodal_tokens, decoder_max_length, args, type='default'):
        self.name = 'medg'
        super().__init__(split, processor, decoder_tokenizer, multimodal_tokens, decoder_max_length, type)

        medg_dir = args.medg_dir
        medg_json_dir = os.path.join(medg_dir, 'json')
        target_annotated_size = get_target_annotated_size(medg_dir)

        json_prefix = {
            'train': 'train_',
            'val': 'minitest_Totalsegmentator',
            'test': 'minitest_medg',
        }[self.split]

        self.imgs, self.qas = [], []
        for filename in tqdm(sorted(os.listdir(medg_json_dir))):
            if (not filename.endswith('.json')) or (not filename.startswith(json_prefix)):
                continue

            with open(os.path.join(medg_json_dir, filename), 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            for filename, pairs in json_data['data'].items():
                image_path = os.path.join(medg_dir, filename)
                for pair in pairs:
                    boxes = sorted(pair['box'], key=lambda box: box[0])
                    boxes = str([[int((v / target_annotated_size) * 1000) for v in box] for box in boxes]).replace(' ', '')
                    self.imgs.append(image_path)
                    self.qas.append((pair['label'], boxes))


class MIMICDataset(BaseDataset):
    def __init__(self, split, processor, decoder_tokenizer, multimodal_tokens, decoder_max_length, args, type='default'):
        self.name = 'mimic'
        super().__init__(split, processor, decoder_tokenizer, multimodal_tokens, decoder_max_length, type)

        json_name = {
            'train': 'chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v1.json',
            'val': 'chat_dev_MIMIC_CXR_all_gpt4extract_rulebased_v1.json',
            'test': 'chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v1.json',
        }[self.split]
        json_path = os.path.join(args.mimic_json_dir, json_name)

        image_dir = args.mimic_image_dir
        target_annotated_size = get_target_annotated_size(args.mimic_image_dir)

        _imgs, _texts, _cxr_labels = read_mimic_img_txt(json_path, args.chexmask_mimic_path, exclude_normal=False, unique_image=False)
        _imgs = [os.path.join(image_dir, im) for im in _imgs]

        mimic_ext_dict = {}
        for _, row in list(pd.read_csv(args.mimic_ext_path).iterrows()):
            image_id = row['image_id']
            boxes = ast.literal_eval(row['bboxes'])
            boxes = convert_pad_space(boxes, (row['width'], row['height']), target_annotated_size)
            boxes.sort(key=lambda box: box[0])
            boxes = str([[int(v * 1000) for v in box] for box in boxes]).replace(' ', '')
            mimic_ext_dict.setdefault(image_id, []).append((row['text'], boxes))

        self.imgs, self.qas, self.cxr_labels = [], [], []
        for image_path, report, cxr_label_dict in zip(_imgs, _texts, _cxr_labels):
            image_id = os.path.basename(image_path).replace('.jpg', '')
            mimic_ext = mimic_ext_dict.get(image_id, None)
            if mimic_ext is None:
                continue
            for ground_text, boxes in mimic_ext:
                self.imgs.append(image_path)
                self.qas.append((ground_text, boxes))
                self.cxr_labels.append(cxr_label_dict)


class PadChestDataset(BaseDataset):
    def __init__(self, split, processor, decoder_tokenizer, multimodal_tokens, decoder_max_length, args, type='default'):
        self.name = 'padchest'
        super().__init__(split, processor, decoder_tokenizer, multimodal_tokens, decoder_max_length, type)

        train_data, test_data = read_padchest_gr(
            args.padchest_grounded_reports_path,
            args.padchest_master_table_path,
            args.padchest_ori_size_path,
            png_to_jpg=True,
            test_split=self.split,
        )
        data_list = {'train': train_data, 'val': test_data, 'test': test_data}[self.split]

        data_dict = OrderedDict()
        for d in data_list:
            data_dict.setdefault(d['ImageID'], []).append(d)

        target_annotated_size = get_target_annotated_size(args.padchest_image_dir)

        self.imgs, self.qas = [], []
        for image_id, data in data_dict.items():
            image_path = os.path.join(args.padchest_image_dir, image_id)
            for elem in data:
                boxes = convert_pad_space(elem['boxes'], elem['ori_size'], target_annotated_size)
                boxes.sort(key=lambda box: box[0])
                boxes = str([[int(v * 1000) for v in box] for box in boxes]).replace(' ', '')
                self.imgs.append(image_path)
                self.qas.append((elem['sentence_en'], boxes))


class TN5000Dataset(BaseDataset):
    def __init__(self, split, processor, decoder_tokenizer, multimodal_tokens, decoder_max_length, args, type='default'):
        self.name = 'tn5000'
        super().__init__(split, processor, decoder_tokenizer, multimodal_tokens, decoder_max_length, type)

        target_annotated_size = get_target_annotated_size(args.tn5000_dir)
        json_name = {'train': 'train.json', 'val': 'val.json', 'test': 'test.json'}[self.split]

        with open(os.path.join(args.tn5000_dir, json_name), 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        data = json_data.get('data', {})
        self.imgs, self.qas = [], []
        for rel_path, pairs in data.items():
            image_path = os.path.join(args.tn5000_dir, rel_path)
            for pair in pairs:
                boxes = sorted(pair['box'], key=lambda box: box[0])
                boxes = str([[int((v / target_annotated_size) * 1000) for v in box] for box in boxes]).replace(' ', '')
                self.imgs.append(image_path)
                self.qas.append((pair['label'], boxes))


class SegTHORDataset(BaseDataset):
    def __init__(self, split, processor, decoder_tokenizer, multimodal_tokens, decoder_max_length, args, type='default'):
        self.name = 'segthor'
        BaseDataset.__init__(self, split, processor, decoder_tokenizer, multimodal_tokens, decoder_max_length, type)

        target_annotated_size = get_target_annotated_size(args.segthor_dir)
        json_name = {'train': 'train.json', 'val': 'val.json', 'test': 'test.json'}[self.split]

        with open(os.path.join(args.segthor_dir, json_name), 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        data = json_data.get('data', {})
        self.imgs, self.qas = [], []
        self.metas = []
        for rel_path, pairs in data.items():
            image_path = os.path.join(args.segthor_dir, rel_path)

            for pair in pairs:
                boxes = sorted(pair['box'], key=lambda box: box[0])
                boxes = str([[int((v / target_annotated_size) * 1000) for v in box] for box in boxes]).replace(' ', '')
                self.imgs.append(image_path)
                self.qas.append((pair['label'], boxes))
                self.metas.append({'image_path': image_path})
