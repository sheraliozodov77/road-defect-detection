"""
remove_irrelavant_classes.py

Removes unwanted classes from a COCO annotation file.

Functionality:
- Deletes annotations whose categories are not in a keep-list.
- Updates the categories section accordingly.
- Saves the cleaned annotation JSON.

Recommended before model training if some defect types are irrelevant.
"""

import json
import os
from tqdm import tqdm

# CONFIG
merged_json_path = "data/merged/merged_coco.json"
image_dir = "data/merged/images"
output_path = "data/merged/merged_coco_cleaned.json"
remove_ids = {2, 4, 8}  # class IDs to remove

# LOAD JSON
with open(merged_json_path, "r") as f:
    coco = json.load(f)

# Step 1: Remove unwanted annotations
filtered_annotations = [a for a in coco["annotations"] if a["category_id"] not in remove_ids]

# Step 2: Get valid image IDs that still have annotations
valid_image_ids = set()
for ann in filtered_annotations:
    valid_image_ids.add(ann["image_id"])

# Step 3: Filter images, and identify orphan images to delete
filtered_images = []
orphan_image_filenames = []

for img in coco["images"]:
    if img["id"] in valid_image_ids:
        filtered_images.append(img)
    else:
        orphan_image_filenames.append(img["file_name"])

# Step 4: Filter categories
filtered_categories = [c for c in coco["categories"] if c["id"] not in remove_ids]

# Step 5: Save cleaned COCO JSON
filtered_coco = {
    "images": filtered_images,
    "annotations": filtered_annotations,
    "categories": filtered_categories
}
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(filtered_coco, f, indent=4)
print(f" Cleaned COCO JSON saved: {output_path}")

# Step 6: Delete orphan image files
deleted = 0
for fname in tqdm(orphan_image_filenames, desc="🧹 Deleting orphan images"):
    img_path = os.path.join(image_dir, fname)
    if os.path.exists(img_path):
        os.remove(img_path)
        deleted += 1

# === Summary ===
print("\n Summary:")
print(f" Total remaining images      : {len(filtered_images)}")
print(f" Total remaining annotations : {len(filtered_annotations)}")
print(f" Total deleted images        : {deleted}")
print(f" Categories kept             : {[c['name'] for c in filtered_categories]}")
