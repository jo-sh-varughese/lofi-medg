import os
import random
import sys

import torch
from PIL import Image

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

from lofi_utils.gemma import apply_chat_template, pad_or_truncate


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, split, processor, decoder_tokenizer, multimodal_tokens, decoder_max_length, type='default'):
        self.type = type
        self.split = split
        self.processor = processor
        self.decoder_tokenizer = decoder_tokenizer

        self.multimodal_tokens = multimodal_tokens
        self.decoder_max_length = decoder_max_length
        self.context = '[multimodal]' * self.multimodal_tokens

        self.begin_pattern = self.decoder_tokenizer.encode('<start_of_turn>user')[1:]
        self.end_token = self.decoder_tokenizer.encode('<end_of_turn>')[-1]
        self.ignore_tokens = [self.decoder_tokenizer.encode('<bos>')[-1], self.decoder_tokenizer.encode('<pad>')[-1], self.decoder_tokenizer.encode('[multimodal]')[-1]]

        print(f'Initializing {self.name} ({self.type}) dataset...')
        self.cxr_labels = None

        # Set by main.py when --feature_cache_dir is given. When present the
        # image is never opened: the frozen encoder's output was computed once
        # by tools/precompute_features.py and is loaded from disk instead.
        self.feature_cache = None

    def _load_image(self, image_path):
        if self.feature_cache is not None:
            return torch.from_numpy(self.feature_cache.load(image_path).astype('float32'))
        img = Image.open(image_path).convert('RGB')
        return self.processor(images=[img], return_tensors='pt')['pixel_values'][0]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.type == 'qa':
            question, answer = self.qas[index]
            return self._load_image(self.imgs[index]), question, answer

        # for MIMIC-CXR
        if self.cxr_labels is not None:
            cxr_label_dict = self.cxr_labels[index]
            if not any(v == 1 for k, v in cxr_label_dict.items() if k != 'No Finding'):
                if random.random() < 0.7:
                    return self.__getitem__(random.randrange(len(self)))

        img, (question, answer) = self.imgs[index], self.qas[index]

        img = self._load_image(img)

        if self.name in ['slake', 'vqarad', 'omnimedvqa']:
            instruction = question
            response = answer
        elif self.name in ['padchest', 'tn5000', 'segthor']:
            question = question[:-1] if question.endswith('.') else question
            instruction = f'Detect all instances of "{question}".'
            response = f'```\n{answer}\n```'
        else:
            is_grounding = random.random() < 0.5
            if is_grounding:
                question = question[:-1] if question.endswith('.') else question
                instruction = f'Detect all instances of "{question}".'
                response = f'```\n{answer}\n```'
            else:
                instruction = f'Describe the regions: {answer}'
                response = question

        prompt = apply_chat_template(f'{self.context}\n{instruction}', response)
        tokens = self.decoder_tokenizer.encode(prompt)
        tokens = pad_or_truncate(tokens, self.decoder_max_length, pad_token_id=0)

        attention_mask = self._build_attention_mask(tokens, self.ignore_tokens, self.begin_pattern, self.end_token)

        return img, torch.tensor(tokens), torch.tensor(attention_mask).long()

    @staticmethod
    def _build_attention_mask(tokens, ignore_tokens, begin_pattern, end_token):
        attention_mask = [1] * len(tokens)
        inside = False
        pt_len = len(begin_pattern)
        for r_i in range(len(tokens)):
            if tokens[r_i] in ignore_tokens:
                attention_mask[r_i] = 0
            if (r_i <= len(tokens) - pt_len) and (tokens[r_i:r_i + pt_len] == begin_pattern):
                inside = True
            if inside:
                attention_mask[r_i] = 0
            if tokens[r_i] == end_token and inside:
                inside = False
        return attention_mask
