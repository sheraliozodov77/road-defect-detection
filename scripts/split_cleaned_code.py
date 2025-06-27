"""
split_cleaned_code.py

Splits a cleaned, merged dataset into train, val, and test subsets.

Functionality:
- Applies stratified or random splitting logic.
- Outputs separate COCO JSON files for each split.
- Optionally copies images to new directories.

Crucial step before training for organizing data correctly.
"""

import os
import json
import random
import shutil
from tqdm import tqdm

# CONFIG
SEED = 42
VAL_RATIO = 0.1
TEST_RATIO = 0.1

INPUT_JSON = "data/merged/merged_coco_cleaned.json"
IMG_DIR = "data/merged/images"
OUTPUT_BASE = "data/merged"
OUTPUT_IMG_DIR = os.path.join(OUTPUT_BASE, "images")
ANNOTATIONS_DIR = os.path.join(OUTPUT_BASE, "annotations")
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

# SETUP SPLIT DIRS
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_IMG_DIR, split), exist_ok=True)

# LOAD COCO
with open(INPUT_JSON) as f:
    coco = json.load(f)

# SPLIT IMAGES
images = coco["images"]
random.seed(SEED)
random.shuffle(images)

n = len(images)
n_val = int(n * VAL_RATIO)
n_test = int(n * TEST_RATIO)
n_train = n - n_val - n_test

splits = {
    "train": images[:n_train],
    "val": images[n_train:n_train + n_val],
    "test": images[n_train + n_val:]
}

# MAP image_id -> split
image_id_to_split = {}
for split, imgs in splits.items():
    for img in imgs:
        image_id_to_split[img["id"]] = split

# SPLIT ANNOTATIONS
split_annotations = {"train": [], "val": [], "test": []}
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    split = image_id_to_split.get(img_id)
    if split:
        split_annotations[split].append(ann)

# GENERATE SPLIT JSONs + COPY IMAGES
for split in ["train", "val", "test"]:
    imgs = splits[split]
    anns = split_annotations[split]

    out_json = {
        "images": imgs,
        "annotations": anns,
        "categories": coco["categories"]
    }

    with open(os.path.join(ANNOTATIONS_DIR, f"{split}_coco.json"), "w") as f:
        json.dump(out_json, f, indent=4)

    for img in tqdm(imgs, desc=f"Copying {split} images"):
        src_path = os.path.join(IMG_DIR, img["file_name"])
        dst_path = os.path.join(OUTPUT_IMG_DIR, split, img["file_name"])
        shutil.copy(src_path, dst_path)

    print(f" {split}: {len(imgs)} images, {len(anns)} annotations → {split}_coco.json")

# === Summary ===
print("\n Split complete:")
print(f"  Train: {n_train} images")
print(f"  Val  : {n_val} images")
print(f"  Test : {n_test} images")