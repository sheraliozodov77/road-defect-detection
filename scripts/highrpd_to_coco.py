"""
highrpd_to_coco.py

Converts HighRPD dataset's annotations to COCO JSON format.

Functionality:
- Parses each entry to create COCO 'images', 'annotations', and 'categories'.
- Outputs a clean and valid COCO file for downstream conversion.

"""

import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "../data/HighRPD/images")
LBL_DIR = os.path.join(BASE_DIR, "../data/HighRPD/labels")
OUT_JSON = os.path.join(BASE_DIR, "../data/HighRPD/highrpd_coco.json")

categories = [
{
    "id": 0,
    "name": "longitudinal_crack"
},
{
    "id": 1,
    "name": "transverse_crack"
},
{
    "id": 2,
    "name": "oblique_crack"
},
{
    "id": 3,
    "name": "alligator_crack"
},
{
    "id": 4,
    "name": "patch"
},
{
    "id": 5,
    "name": "pothole"
},
{
    "id": 6,
    "name": "line_crack"
},
{
    "id": 7,
    "name": "block_crack"
}
]

def yolo_to_coco(images_dir, labels_dir, output_json):
    coco = {"images": [], "annotations": [], "categories": categories}
    ann_id = 0
    for img_id, fname in enumerate(sorted(os.listdir(images_dir))):
        if not fname.endswith(".jpg"): continue
        img_path = os.path.join(images_dir, fname)
        lbl_path = os.path.join(labels_dir, fname.replace(".jpg", ".txt"))
        if not os.path.exists(lbl_path): continue

        coco["images"].append({
            "id": img_id,
            "file_name": fname,
            "width": 640,
            "height": 640
        })

        with open(lbl_path, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) != 5:
                    continue
                class_id, xc, yc, w, h = parts
                x = (xc - w / 2) * 640
                y = (yc - h / 2) * 640
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(class_id) + 6,
                    "bbox": [x, y, w * 640, h * 640],
                    "area": w * h * 640 * 640,
                    "iscrowd": 0
                })
                ann_id += 1

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=4)
    print(f"COCO JSON created at: {output_json}")

if __name__ == "__main__":
    yolo_to_coco(IMG_DIR, LBL_DIR, OUT_JSON)
