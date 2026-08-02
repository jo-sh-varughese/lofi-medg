import json
import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

from lofi_utils.dataset.base import BaseDataset


class SLAKEDataset(BaseDataset):
    def __init__(self, split, processor, decoder_tokenizer, multimodal_tokens, decoder_max_length, args, type='default'):
        self.name = 'slake'
        super().__init__(split, processor, decoder_tokenizer, multimodal_tokens, decoder_max_length, type)

        slake_dir = args.slake_dir
        img_dir = os.path.join(slake_dir, 'imgs')
        json_name = {'train': 'train.json', 'val': 'validation.json', 'test': 'test.json'}[self.split]
        json_path = os.path.join(slake_dir, json_name)

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        self.imgs, self.qas = [], []
        self.metas = []
        for sample in json_data:
            if sample['q_lang'] != 'en':
                continue
            img_name = sample['img_name']
            question = sample['question']
            answer = sample['answer']
            self.imgs.append(os.path.join(img_dir, img_name))
            self.qas.append((question, answer))

            sample = dict(sample)
            self.metas.append(sample)


class VQARADDataset(BaseDataset):
    def __init__(self, split, processor, decoder_tokenizer, multimodal_tokens, decoder_max_length, args, type='default'):
        self.name = 'vqarad'
        super().__init__(split, processor, decoder_tokenizer, multimodal_tokens, decoder_max_length, type)

        vqarad_dir = args.vqarad_dir
        with open(os.path.join(vqarad_dir, 'VQA_RAD Dataset Public.json'), 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        img_dir = os.path.join(vqarad_dir, 'VQA_RAD Image Folder')
        split_df = pd.read_csv(args.vqarad_split_tsv_path, sep='\t')
        qid_to_split = {str(r['QID_unique']).strip(): str(r['SPLIT_BALANCED']).strip().lower() for _, r in split_df.iterrows()}

        self.imgs, self.qas = [], []
        self.metas = []
        for sample in json_data:
            qid = str(sample.get('qid', '')).strip()
            split_name = qid_to_split.get(qid, None)
            if split_name is None:
                continue
            include = split_name == {'train': 'train', 'val': 'validation', 'test': 'test'}[self.split]
            if not include:
                continue

            question = self._capitalize_first_char(str(sample['question']))
            answer = self._capitalize_first_char(str(sample['answer'])).strip().rstrip('.')
            if answer not in {'Yes', 'No'}:
                continue

            self.imgs.append(os.path.join(img_dir, sample['image_name']))
            self.qas.append((question, answer))
            sample = dict(sample)
            sample['question'] = question
            sample['answer'] = answer
            sample['answer_type'] = str(sample['answer_type']).strip().upper()
            self.metas.append(sample)

    @staticmethod
    def _capitalize_first_char(text):
        text = str(text)
        if len(text) == 0:
            return text
        return text[:1].upper() + text[1:]


class OmniMedVQADataset(BaseDataset):
    def __init__(self, split, processor, decoder_tokenizer, multimodal_tokens, decoder_max_length, args, type='default'):
        self.name = 'omnimedvqa'
        super().__init__(split, processor, decoder_tokenizer, multimodal_tokens, decoder_max_length, type)

        json_name = {'train': 'train.json', 'val': 'val.json', 'test': 'test.json'}[self.split]

        with open(os.path.join(args.omnimedvqa_dir, json_name), 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        self.imgs, self.qas = [], []
        self.metas = []
        for rel_path, qa_list in json_data['data'].items():
            image_path = os.path.join(args.omnimedvqa_dir, rel_path)
            for qa in qa_list:
                question = str(qa['question']).strip()
                answer = str(qa['answer']).strip()
                self.imgs.append(image_path)
                self.qas.append((question, answer))
                self.metas.append({'answer_type': 'OPEN'})
