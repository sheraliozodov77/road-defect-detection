"""
merge_coco_and_images.py

Merges multiple datasets (images and COCO annotations) into one unified dataset.

Functionality:
- Updates image/annotation IDs to avoid collisions.
- Copies and renames image files into merged folders.
- Merges COCO JSONs into a single file split into train/val/test.

Use this script before final conversion to YOLO format.
"""

import os
import json
import shutil
from tqdm import tqdm

# CONFIG
DATASETS = [
    {"json": "data/UAV_PDD2023/uavpdd2023_coco.json", "img_dir": "data/UAV_PDD2023/images"},
    {"json": "data/HighRPD/highrpd_coco.json", "img_dir": "data/HighRPD/images"},
    {"json": "data/RDD2022/rdd2022_japan_coco.json", "img_dir": "data/RDD2022/RDD2022_Japan/images"},
    {"json": "data/RDD2022/rdd2022_india_coco.json", "img_dir": "data/RDD2022/RDD2022_India/images"},
    {"json": "data/RDD2022/rdd2022_china_motorbike_coco.json", "img_dir": "data/RDD2022/RDD2022_China_MotorBike/images"},
    {"json": "data/RDD2022/rdd2022_china_drone_coco.json", "img_dir": "data/RDD2022/RDD2022_China_Drone/images"},
]

MERGED_IMG_DIR = "data/merged/images"
MERGED_JSON_PATH = "data/merged/merged_coco.json"
os.makedirs(MERGED_IMG_DIR, exist_ok=True)

# Merge Logic
merged = {"images": [], "annotations": [], "categories": []}
seen_categories = {}
img_id, ann_id = 0, 0
filename_map = {}

# Register all categories first
for ds in DATASETS:
    with open(ds["json"], "r") as f:
        coco = json.load(f)
        for cat in coco["categories"]:
            if cat["id"] not in seen_categories:
                seen_categories[cat["id"]] = cat
merged["categories"] = list(seen_categories.values())

# Merge images and annotations
for ds in DATASETS:
    with open(ds["json"], "r") as f:
        coco = json.load(f)

    image_dir = ds["img_dir"]

    for image in tqdm(coco["images"], desc=f"Processing {os.path.basename(ds['json'])}"):
        new_filename = f"{img_id}_{image['file_name']}"
        src_path = os.path.join(image_dir, image["file_name"])
        dst_path = os.path.join(MERGED_IMG_DIR, new_filename)

        if not os.path.exists(src_path):
            print(f" Missing: {src_path}")
            continue

        shutil.copy(src_path, dst_path)
        filename_map[image["id"]] = img_id

        merged["images"].append({
            "id": img_id,
            "file_name": new_filename,
            "width": image["width"],
            "height": image["height"]
        })
        img_id += 1

    for ann in coco["annotations"]:
        old_id = ann["image_id"]
        if old_id not in filename_map:
            continue
        merged["annotations"].append({
            "id": ann_id,
            "image_id": filename_map[old_id],
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
            "area": ann["area"],
            "iscrowd": ann.get("iscrowd", 0)
        })
        ann_id += 1

# Save Final COCO JSON
with open(MERGED_JSON_PATH, "w") as f:
    json.dump(merged, f, indent=4)

print(f"\n Merged COCO saved to: {MERGED_JSON_PATH}")
print(f"\n Total images: {len(merged['images'])}")
print(f"\n Total annotations: {len(merged['annotations'])}")