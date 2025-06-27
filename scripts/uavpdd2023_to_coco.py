"""
uavpdd2023_to_coco.py

Converts UAV-PDD2023 dataset from Pascal VOC XML to COCO JSON format.

Functionality:
- Extracts object annotations from XML.
- Maps distress types to COCO category IDs.
- Outputs compatible JSON for training or conversion.

Run this after downloading UAV-PDD2023 to normalize it.
"""

import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_NAME = "UAV_PDD2023"
XML_DIR = os.path.join(BASE_DIR, f"../data/{DATASET_NAME}/Annotations")
IMG_DIR = os.path.join(BASE_DIR, f"../data/{DATASET_NAME}/JPEGImages")
OUT_JSON = os.path.join(BASE_DIR, f"../data/{DATASET_NAME}/uavpdd2023_coco.json")

UNIFIED_LABELS = {
    "Longitudinal crack": 0,
    "Transverse crack": 1,
    "Oblique crack": 2,
    "Alligator crack": 3,
    "Repair": 4,
    "Pothole": 5
}

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

def convert(xml_dir, out_json, img_dir):
    coco = {"images": [], "annotations": [], "categories": categories}
    ann_id = 0
    image_id = 0

    for xml_file in tqdm(sorted(os.listdir(xml_dir))):
        if not xml_file.endswith(".xml"):
            continue

        tree = ET.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()

        filename = root.find("filename").text
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        coco["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        for obj in root.findall("object"):
            label = obj.find("name").text
            if label not in UNIFIED_LABELS:
                continue

            category_id = UNIFIED_LABELS[label]
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "area": (xmax - xmin) * (ymax - ymin),
                "iscrowd": 0
            })
            ann_id += 1

        image_id += 1

    with open(out_json, "w") as f:
        json.dump(coco, f, indent=4)
    print(f"Converted to COCO: {out_json}")

if __name__ == "__main__":
    convert(XML_DIR, OUT_JSON, IMG_DIR)
