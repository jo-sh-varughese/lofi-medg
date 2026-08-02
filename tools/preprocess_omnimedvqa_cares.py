import argparse
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from tools.preprocess_utils import resize_with_padding, save_jpg

CARES_OMNIMEDVQA_URL = (
    "https://raw.githubusercontent.com/richard-peng-xia/CARES/refs/heads/main/data/"
    "OmniMedVQA/omnimedvqa_factuality.jsonl"
)
CARES_JSONL_NAME = "omnimedvqa_factuality.jsonl"
SEED = 0
SPLITS = ("train", "val", "test")


def _capitalize_first_char(text):
    text = str(text)
    if len(text) == 0:
        return text
    return text[:1].upper() + text[1:]


def normalize_question_answer(question, answer):
    question = _capitalize_first_char((question or "").strip())
    answer = _capitalize_first_char((answer or "").strip()).rstrip(".")
    return question, answer


def download_jsonl_to_cwd(url):
    save_path = os.path.join(os.getcwd(), CARES_JSONL_NAME)
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)
    return save_path


def read_jsonl(path):
    rows = []
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_rows_inplace(rows):
    for row in rows:
        question, answer = normalize_question_answer(row.get("text"), row.get("answer"))
        row["text"] = question
        row["answer"] = answer


def filter_rows_with_existing_dataset_dirs(rows, omnimedvqa_dir):
    images_root = Path(omnimedvqa_dir) / "Images"
    datasets_in_rows = sorted({row.get("dataset") for row in rows if row.get("dataset")})

    existing_datasets = set()
    for dataset in datasets_in_rows:
        dataset_dir = images_root / dataset
        if dataset_dir.is_dir():
            existing_datasets.add(dataset)

    filtered_rows = [row for row in rows if row.get("dataset") in existing_datasets]
    return filtered_rows


def get_multi_answer_question_set(rows):
    question_to_answers = {}
    for row in rows:
        question = row.get("text", "")
        answer = row.get("answer", "")
        if not question or not answer:
            continue
        if question not in question_to_answers:
            question_to_answers[question] = set()
        question_to_answers[question].add(answer)
    return {q for q, answers in question_to_answers.items() if len(answers) > 1}


def extract_patient_id(row):
    dataset = row.get("dataset")
    image = row.get("image")
    if not dataset or not image:
        return None
    parts = Path(image).parts
    if len(parts) < 4:
        return None
    if parts[0] != "Images" or parts[1] != dataset:
        return None
    patient_parts = parts[2:-1]
    if not patient_parts:
        return None
    return "/".join(patient_parts)


def split_ids_80_10_10(id_list, rng):
    ids = list(id_list)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:n_train + n_val + n_test])
    return train_ids, val_ids, test_ids


def assign_splits(rows, seed=SEED):
    rng = random.Random(seed)
    split_map = {}
    patient_level_datasets = {"DeepDRiD", "RUS CHN"}

    # patient-level split for specific datasets
    for dataset_name in patient_level_datasets:
        ds_rows = [row for row in rows if row.get("dataset") == dataset_name]
        ds_row_pid = [(row, extract_patient_id(row)) for row in ds_rows]
        patient_ids = sorted({pid for _, pid in ds_row_pid if pid})
        train_ids, val_ids, test_ids = split_ids_80_10_10(patient_ids, rng)
        for row, pid in ds_row_pid:
            if pid in train_ids:
                split_map[id(row)] = "train"
            elif pid in val_ids:
                split_map[id(row)] = "val"
            elif pid in test_ids:
                split_map[id(row)] = "test"

    # sample-level split for all remaining datasets
    remaining_rows = [row for row in rows if id(row) not in split_map]
    indices = list(range(len(remaining_rows)))
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_idx = set(indices[:n_train])
    val_idx = set(indices[n_train:n_train + n_val])
    for i, row in enumerate(remaining_rows):
        if i in train_idx:
            split_map[id(row)] = "train"
        elif i in val_idx:
            split_map[id(row)] = "val"
        else:
            split_map[id(row)] = "test"

    return split_map


def build_and_save_dataset(rows, split_map, args):
    os.makedirs(args.output_dir, exist_ok=True)
    split_json = {
        "train": {"modality": "", "data": {}},
        "val": {"modality": "", "data": {}},
        "test": {"modality": "", "data": {}},
    }
    for split in SPLITS:
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)

    for row in tqdm(rows, desc="preprocess omnimedvqa"):
        split = split_map.get(id(row), None)
        if split is None:
            continue

        dataset = row.get("dataset", "unknown")
        question = row.get("text", "")
        answer = row.get("answer", "")
        image_rel = row.get("image", "")
        if not question or not answer or not image_rel:
            continue

        image_src = Path(args.omnimedvqa_dir) / image_rel
        if not image_src.is_file():
            continue

        image_no_prefix = image_rel
        if image_no_prefix.startswith("Images/"):
            image_no_prefix = image_no_prefix[len("Images/"):]
        image_no_ext = os.path.splitext(image_no_prefix)[0]
        out_name = f"{image_no_ext.replace('/', '-')}.jpg"
        rel_path = f"{split}/{out_name}"
        out_path = Path(args.output_dir) / rel_path

        if not out_path.exists():
            image = Image.open(image_src).convert("RGB")
            save_jpg(resize_with_padding(image, args.target_size), str(out_path))

        if rel_path not in split_json[split]["data"]:
            split_json[split]["data"][rel_path] = []
        split_json[split]["data"][rel_path].append(
            {
                "question": question,
                "answer": answer,
                "dataset": dataset,
                "question_id": row.get("question_id", ""),
            }
        )

    for split in SPLITS:
        json_path = os.path.join(args.output_dir, f"{split}.json")
        with open(json_path, "wt", encoding="utf-8") as f:
            json.dump(split_json[split], f, indent=2, ensure_ascii=False)
        num_images = len(split_json[split]["data"])
        num_samples = sum(len(v) for v in split_json[split]["data"].values())
        print(f"{split}: images={num_images}, samples={num_samples}, json={json_path}")


def run(args):
    jsonl_path = download_jsonl_to_cwd(CARES_OMNIMEDVQA_URL)
    rows = read_jsonl(jsonl_path)
    normalize_rows_inplace(rows)
    print(f"Downloaded JSONL: {jsonl_path}")
    print(f"Loaded rows: {len(rows)}")

    rows = filter_rows_with_existing_dataset_dirs(rows, args.omnimedvqa_dir)
    multi_answer_question_set = get_multi_answer_question_set(rows)
    rows = [
        row
        for row in rows
        if row.get("text", "") in multi_answer_question_set
    ]
    print(f"Rows used for split/build (multi-answer questions only): {len(rows)}")

    split_map = assign_splits(rows, seed=SEED)
    build_and_save_dataset(rows, split_map, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--omnimedvqa_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--target_size", type=int, default=512)
    parser.set_defaults(
        omnimedvqa_dir="../data/OmniMedVQA/",
        output_dir="../data/omnimedvqa_512p/",
    )
    args = parser.parse_args()
    run(args)
