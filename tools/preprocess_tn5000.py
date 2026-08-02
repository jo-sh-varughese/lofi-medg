import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from lofi_utils.misc import convert_pad_space
from tools.preprocess_utils import resize_with_padding, save_jpg

TN5000_LABELS = {
    "0": "benign nodule",
    "1": "malignant nodule",
}


def read_split_ids(imagesets_dir, split):
    split_path = os.path.join(imagesets_dir, "Main", f"{split}.txt")
    with open(split_path, "rt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_voc_boxes(xml_path, area_threshold, target_size):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    image_area = float(width * height)
    min_area = image_area * (area_threshold ** 2)

    boxes_list = []
    for obj in root.findall("object"):
        label = TN5000_LABELS[obj.findtext("name").strip()]

        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        xmin = max(0.0, min(xmin, width))
        xmax = max(0.0, min(xmax, width))
        ymin = max(0.0, min(ymin, height))
        ymax = max(0.0, min(ymax, height))
        if xmax <= xmin or ymax <= ymin:
            continue

        area = (xmax - xmin) * (ymax - ymin)
        if area < min_area:
            continue

        boxes = [[xmin / width, ymin / height, xmax / width, ymax / height]]
        boxes = convert_pad_space(boxes, (width, height), size=target_size)
        boxes = [[int(v * target_size) for v in box] for box in boxes]
        boxes_list.append({"box": boxes, "label": label})

    return boxes_list


def preprocess_split(split, split_ids, args):
    save_dict = {}
    save_dir = os.path.join(args.output_dir, split)
    os.makedirs(save_dir, exist_ok=True)

    for sample_id in tqdm(split_ids, desc=f"preprocess {split}"):
        image_path = os.path.join(args.tn5000_dir, "JPEGImages", f"{sample_id}.jpg")
        xml_path = os.path.join(args.tn5000_dir, "Annotations", f"{sample_id}.xml")
        if (not os.path.exists(image_path)) or (not os.path.exists(xml_path)):
            continue

        boxes_list = parse_voc_boxes(xml_path, args.area_threshold, args.target_size)
        if len(boxes_list) == 0:
            continue

        image = Image.open(image_path).convert("RGB")
        out_name = f"{sample_id}.jpg"
        out_path = os.path.join(save_dir, out_name)
        save_jpg(resize_with_padding(image, args.target_size), out_path)
        save_dict[f"{split}/{out_name}"] = boxes_list

    return save_dict


def run(args):
    imagesets_dir = os.path.join(args.tn5000_dir, "ImageSets")
    split_to_ids = {
        "train": read_split_ids(imagesets_dir, "train"),
        "val": read_split_ids(imagesets_dir, "val"),
        "test": read_split_ids(imagesets_dir, "test"),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    split_json = {
        "train": {"modality": "US", "data": {}},
        "val": {"modality": "US", "data": {}},
        "test": {"modality": "US", "data": {}},
    }

    for split, split_ids in split_to_ids.items():
        split_json[split]["data"] = preprocess_split(split, split_ids, args)
        json_path = os.path.join(args.output_dir, f"{split}.json")
        with open(json_path, "wt", encoding="utf-8") as f:
            json.dump(split_json[split], f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tn5000_dir",
        type=str,
        default="../data/TN5000/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/tn5000_512p/",
    )
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--area_threshold", type=float, default=0.01)
    ARGS = parser.parse_args()
    run(ARGS)
