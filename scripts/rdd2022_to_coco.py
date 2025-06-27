"""
rdd2022_to_coco.py

Converts RDD2022 dataset annotations from Pascal VOC XML format to COCO JSON format.

Functionality:
- Parses XML files for bounding boxes and class names.
- Normalizes fields and aggregates into COCO-style JSON.
- Supports multiple countries and regions in RDD2022.

Run this to standardize RDD2022 for use in object detection pipelines.
"""

import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

COUNTRY = "China_Drone"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_DIR = os.path.join(BASE_DIR, f"../data/RDD2022/RDD2022_{COUNTRY}/annotations/xmls")
IMG_DIR = os.path.join(BASE_DIR, f"../data/RDD2022/RDD2022_{COUNTRY}/images")
OUT_PATH = os.path.join(BASE_DIR, f"../data/RDD2022/rdd2022_{COUNTRY.lower()}_coco.json")

DAMAGE_LABELS = {
    "D00": 0,
    "D10": 1,
    "D20": 3,
    "D40": 5
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

def convert_voc_to_coco(xml_dir, output_json, img_dir):
    coco = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
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
            if label not in DAMAGE_LABELS:
                continue

            category_id = DAMAGE_LABELS[label]
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            bbox_width = xmax - xmin
            bbox_height = ymax - ymin
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, bbox_width, bbox_height],
                "area": bbox_width * bbox_height,
                "iscrowd": 0
            })
            ann_id += 1

        image_id += 1

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=4)
    print(f"COCO JSON saved to: {output_json}")

if __name__ == "__main__":
    convert_voc_to_coco(XML_DIR, OUT_PATH, IMG_DIR)
