"""
check_class_distribution.py

Visualizes the distribution of defect classes in YOLO label format across a dataset.

Functionality:
- Scans a specified labels directory.
- Counts occurrences of each class label.

Usage:
Run this script to verify class balance after annotation or merging.
"""

import json
from collections import Counter

# Path to COCO JSON
COCO_JSON_PATH = "data/merged/merged_coco_cleaned.json"

def load_coco_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_category_mapping(categories):
    """Returns a dictionary {id: name}"""
    return {cat['id']: cat['name'] for cat in categories}

def count_annotations(annotations):
    """Counts occurrences of each category_id in annotations"""
    return Counter([a['category_id'] for a in annotations])

def print_class_distribution(data):
    cat_map = get_category_mapping(data['categories'])
    counts = count_annotations(data['annotations'])

    print("\n Class Distribution:")
    for cat_id, count in sorted(counts.items()):
        name = cat_map.get(cat_id, "Unknown")
        print(f"  {cat_id:2d} - {name:20s}: {count} instances")
    
    missing = set(cat_map.keys()) - set(counts.keys())
    if missing:
        print("\n Warning: The following classes are defined but unused:")
        for m in missing:
            print(f"  {m:2d} - {cat_map[m]}")

if __name__ == "__main__":
    data = load_coco_annotations(COCO_JSON_PATH)
    print_class_distribution(data)
