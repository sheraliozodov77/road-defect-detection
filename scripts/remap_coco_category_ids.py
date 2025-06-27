"""
remap_coco_category_ids.py

Reassigns category IDs in a COCO JSON annotation file to match a new class list.

Functionality:
- Maps existing category names to new integer IDs.
- Outputs a rewritten COCO annotation JSON.

Use this after merging datasets to ensure consistent class indexing.
"""

import os
import json

# Class ID remapping
id_remap = {
    0: 0,  # longitudinal_crack
    1: 1,  # transverse_crack
    3: 2,  # alligator_crack
    5: 3,  # pothole
    6: 4,  # line_crack
    7: 5   # block_crack
}

# New categories based on new IDs
new_categories = [
    {"id": 0, "name": "longitudinal_crack"},
    {"id": 1, "name": "transverse_crack"},
    {"id": 2, "name": "alligator_crack"},
    {"id": 3, "name": "pothole"},
    {"id": 4, "name": "line_crack"},
    {"id": 5, "name": "block_crack"}
]

# Input/output paths
splits = ["train", "val", "test"]
input_dir = "data/merged/annotations"
output_dir = "data/merged/annotations"
os.makedirs(output_dir, exist_ok=True)

for split in splits:
    input_path = os.path.join(input_dir, f"{split}_coco.json")
    output_path = os.path.join(output_dir, f"{split}_coco_reindexed.json")

    with open(input_path, "r") as f:
        coco = json.load(f)

    # Remap annotations
    new_annotations = []
    for ann in coco["annotations"]:
        old_id = ann["category_id"]
        if old_id in id_remap:
            ann["category_id"] = id_remap[old_id]
            new_annotations.append(ann)

    # Assemble new COCO
    new_coco = {
        "images": coco["images"],
        "annotations": new_annotations,
        "categories": new_categories
    }

    with open(output_path, "w") as f:
        json.dump(new_coco, f, indent=2)

    print(f" Saved: {output_path} ({len(new_annotations)} annotations)")
