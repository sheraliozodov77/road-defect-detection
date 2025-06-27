# Road Defect Detection using YOLOv8

This project involves building a YOLOv8-based object detection pipeline to detect various types of road defects across datasets from multiple domains. It follows a full pipeline — from raw annotations to training and evaluation on a GPU-backed environment.

---

## Dataset Description

This project uses a merged dataset composed of three public datasets for road surface distress detection:

### 1. HighRPD (Primary – UAV Imagery)
- **Source**: [Data in Brief - DOI:10.1016/j.dib.2025.111377](https://doi.org/10.1016/j.dib.2025.111377)
- **Size**: 11,696 images (640×640 resolution)
- **Classes**: line cracks, block cracks, potholes
- **Format**: YOLO

### 2. RDD2022 (Secondary – Global Road Data)
- **Source**: [RDD2022 GitHub Repository](https://github.com/sekilab/RoadDamageDetector)
- **Size**: 47,420 images from China, Japan, India, etc.
- **Classes**: longitudinal cracks, transverse cracks, alligator cracks, potholes
- **Format**: VOC XML (converted to COCO, then YOLO)
- **Subsets Used**: China (drone + motorbike), India, Japan

### 3. UAV-PDD2023 (Tertiary – Diverse UAV Dataset)
- **Source**: [Zenodo DOI:10.5281/zenodo.8429208](https://zenodo.org/record/8429208)
- **Size**: 2,440 images (2592×1944 resolution), ~11,158 labels
- **Classes**: longitudinal, transverse, oblique, alligator cracks, patching, potholes
- **Format**: VOC XML (converted to COCO, then YOLO)

All datasets were normalized and unified into YOLO format. The merged dataset was split into training, validation, and test sets.

---

## Annotation Conversion Pipeline

The following steps were used to unify the format:

1. **Convert XML → COCO JSON** using `*_to_coco.py` scripts per dataset.
2. **COCO JSON → YOLO TXT** format using `convert_coco_to_yolo.py`.
3. **Re-map/clean class IDs** and remove irrelevant categories.
4. **Merge datasets** with `merge_coco_and_images.py`.
5. **Split final merged dataset** into `train`, `val`, and `test` folders (80/10/10).
6. YOLO-ready labels are saved under `merged/labels/`, images under `merged/images/`, and converted COCO annotations under `merged/annotations/`.

---

## Training Pipeline

We used [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and trained on Kaggle with T4 GPU.

### Model Configuration:

- Model: `yolov8s.pt`
- Epochs: `20`
- Batch size: `32`
- Image size: `640x640`
- Optimizer: SGD
- Early stopping: patience = 10
- Learning rate: 0.005 → cosine schedule
- Advanced Augmentations:
  - `mixup=0.1`, `cutmix=0.1`
  - `erasing=0.5`, `hsv_s=0.7`, `translate=0.2`, `mosaic=1.0`

### Classes:

1. longitudinal_crack  
2. transverse_crack  
3. alligator_crack  
4. pothole  
5. line_crack  
6. block_crack

---

## Evaluation Results

### Test Set

| Metric       | Value  |
|--------------|--------|
| Precision    | 0.550  |
| Recall       | 0.488  |
| mAP@0.5      | 0.500  |
| mAP@0.5:0.95 | 0.255  |

| Class             | Prec. | Recall | mAP@0.5 | mAP@0.5:0.95 |
|------------------|-------|--------|--------|--------------|
| longitudinal     | 0.561 | 0.569  | 0.556  | 0.290        |
| transverse       | 0.532 | 0.480  | 0.495  | 0.226        |
| alligator        | 0.643 | 0.634  | 0.667  | 0.358        |
| pothole          | 0.563 | 0.453  | 0.496  | 0.222        |
| line             | 0.492 | 0.332  | 0.336  | 0.191        |
| block            | 0.506 | 0.458  | 0.448  | 0.244        |

---

## Output Assets

- Best model weights: `outputs/best_yolov8s.pt`

---

## Challenges Faced

- **Finding Relevant Datasets**: It was challenging to discover UAV-based road defect datasets with consistent annotations.
- **Merging Multiple Datasets**: Combined HighRPD, RDD2022, and UAV-PDD2023, each in a different annotation format (COCO, VOC, or txt).
- **Annotation Conversion**: Required writing scripts to convert XML to COCO and then to YOLO format.
- **GPU Resource Access**: Leveraged Kaggle’s free T4 GPU for training, which required careful tuning of batch size and training time.

---

## Future Work

- Try `yolov8m.pt` or `yolov8l.pt` for better recall.
- Improve class separation using deeper augmentations.

---

## Environment

- Framework: YOLOv8 by Ultralytics
- Training Platform: Kaggle (GPU: NVIDIA Tesla T4)