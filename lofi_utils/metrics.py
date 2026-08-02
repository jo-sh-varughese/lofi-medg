import pickle
import re
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd

CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
MANUAL_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
ARTICLES = ["a", "an", "the"]
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
PUNCT = [
    ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_",
    "-", ">", "<", "@", "`", ",", "?", "!",
]


def normalize_word(token):
    token = str(token)
    normalized = token
    for p in PUNCT:
        if (p + " " in token or " " + p in token) or (re.search(COMMA_STRIP, token) is not None):
            normalized = normalized.replace(p, "")
        else:
            normalized = normalized.replace(p, " ")
    token = PERIOD_STRIP.sub("", normalized, re.UNICODE)

    words = []
    for word in token.lower().split():
        word = MANUAL_MAP.setdefault(word, word)
        if word not in ARTICLES:
            words.append(word)
    for i, word in enumerate(words):
        if word in CONTRACTIONS:
            words[i] = CONTRACTIONS[word]
    return " ".join(words).replace(",", "")


def split_sentence(sentence, n):
    words = defaultdict(int)
    tokens = str(sentence).lower().strip().split()
    for i in range(len(tokens) - n + 1):
        ngram = " ".join(tokens[i:i + n])
        if ngram:
            words[ngram] += 1
    return words


def calculate_exactmatch(candidate, reference):
    candidate_words = split_sentence(normalize_word(candidate), 1)
    reference_words = split_sentence(normalize_word(reference), 1)
    total = sum(candidate_words.values())
    if total == 0:
        return 0.0
    count = sum(1 for word in reference_words if word in candidate_words)
    return count / total


def calc_exactmatch_accuracy(gt_list, gen_list):
    # https://github.com/LLaVA-VL/LLaVA-Med-preview/blob/399ccffdf39924c7f41cc6db44b90577fa9eea0b/llava/eval/eval_metrics/evaluate_metrics.py
    if len(gt_list) == 0:
        return 0.0
    return sum(calculate_exactmatch(gen, gt) for gt, gen in zip(gt_list, gen_list)) / len(gt_list)


def box_iou_xyxy(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)

    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / np.maximum(union, 1e-12)


def rex_omni_pr_re_f1(predictions, targets, iou_thr=0.5):
    # https://raw.githubusercontent.com/IDEA-Research/Rex-Omni/refs/heads/master/evaluation/metrics/other_metric.py
    total_gt = 0
    total_pred = 0
    matches = 0

    for (p_boxes, p_cls, _), (g_boxes, g_cls) in zip(predictions, targets):
        p_boxes = np.asarray(p_boxes, np.float32).reshape(-1, 4)
        g_boxes = np.asarray(g_boxes, np.float32).reshape(-1, 4)
        p_cls = np.asarray(p_cls, np.int64)
        g_cls = np.asarray(g_cls, np.int64)

        total_pred += len(p_boxes)  # len(np.zeros((0, 4))) == 0
        total_gt += len(g_boxes)

        if len(p_boxes) == 0 or len(g_boxes) == 0:
            continue

        ious = box_iou_xyxy(p_boxes, g_boxes)

        used_preds = set()
        for j in range(len(g_boxes)):
            best_iou = 0.0
            best_idx = -1

            for i in range(len(p_boxes)):
                if i in used_preds or p_cls[i] != g_cls[j]:
                    continue

                iou = ious[i, j]
                if iou >= iou_thr and iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx != -1:
                matches += 1
                used_preds.add(best_idx)

    if total_gt == 0:
        if total_pred == 0:
            p = r = f1 = 1.0
        else:
            p = r = f1 = 0.0
    else:
        p = matches / total_pred if total_pred > 0 else 0.0
        r = matches / total_gt if total_gt > 0 else 0.0
        f1 = 2 * p * r / (p + r + 1e-12) if (p + r) > 0 else 0.0

    t = int(iou_thr * 100)
    return OrderedDict([
        (f'pr_{t}', float(p)),
        (f're_{t}', float(r)),
        (f'f1_{t}', float(f1)),
    ])


def calc_detection_metrics(pkl_path, save_csv_path, label_list=['0', '1'], mono_class=False):
    pkl = pickle.load(open(pkl_path, 'rb'))
    predictions = pkl['predictions']
    targets = pkl['targets']

    if mono_class:
        _predictions, _targets = [], []
        for (pred_boxes, pred_classes, pred_confidences), (tgt_boxes, tgt_class) in zip(predictions, targets):
            pred_classes[pred_classes != 0] = 1
            tgt_class[tgt_class != 0] = 1
            _predictions.append((pred_boxes, pred_classes, pred_confidences))
            _targets.append((tgt_boxes, tgt_class))
        predictions, targets = _predictions, _targets  # reset

        label_list = ['0', '1']  # reset label_list

    rows = {}

    # calculate precision, recall, and f1
    score_det = rex_omni_pr_re_f1(predictions, targets)
    for key, val in score_det.items():
        print(f'{key}: {val * 100}')
        rows[key] = [val * 100]

    # save csv
    pd.DataFrame(rows).to_csv(save_csv_path, index=False)
