"""
convert_coco_to_yolo.py

Converts COCO-format annotation JSON files to YOLOv8 format.

Functionality:
- Reads a COCO JSON file containing image and annotation data.
- Extracts bounding boxes and remaps category IDs.
- Saves YOLO-style '.txt' files into train/val/test label folders.

Use this script after generating or receiving COCO-formatted labels.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm

def convert_annotations(coco_json_path, output_dir):
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    category_id_to_index = {cat['id']: idx for idx, cat in enumerate(data['categories'])}

    # Create output folder
    label_output_dir = Path(output_dir)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare label dict
    label_dict = {img['id']: [] for img in data['images']}

    for ann in tqdm(data['annotations'], desc=f"Processing {Path(coco_json_path).name}"):
        img_id = ann['image_id']
        bbox = ann['bbox']  # [x, y, width, height]
        category_id = ann['category_id']

        x, y, w, h = bbox
        x_center = x + w / 2
        y_center = y + h / 2

        img_info = next(img for img in data['images'] if img['id'] == img_id)
        img_w, img_h = img_info['width'], img_info['height']

        x_center /= img_w
        y_center /= img_h
        w /= img_w
        h /= img_h

        class_id = category_id_to_index[category_id]
        label_dict[img_id].append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    for img_id, lines in label_dict.items():
        filename = Path(image_id_to_filename[img_id]).stem + ".txt"
        with open(label_output_dir / filename, 'w') as f:
            f.write("\n".join(lines))

if __name__ == "__main__":
    base_path = "data/merged/annotations"
    out_base = "data/merged/labels"

    splits = {
        "train": "train_coco_reindexed.json",
        "val": "val_coco_reindexed.json",
        "test": "test_coco_reindexed.json"
    }

    for split, filename in splits.items():
        coco_json_path = os.path.join(base_path, filename)
        output_dir = os.path.join(out_base, split)
        convert_annotations(coco_json_path, output_dir)

    print(" All annotations converted to YOLO format.")