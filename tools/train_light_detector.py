import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
import numpy as np


def create_dataset_yaml(data_dir: str, output_yaml: str):
    """Generate YOLO dataset YAML config for single-class traffic light detection."""
    yaml_content = f"""# Traffic Light Detection Dataset
path: {os.path.abspath(data_dir)}
train: images/train
val: images/val
nc: 1
names:
  0: traffic_light
"""
    with open(output_yaml, "w") as f:
        f.write(yaml_content.strip())
    print(f"  Dataset configuration created at: {output_yaml}")
    return output_yaml


def convert_lisa_to_yolo(lisa_dir: str, output_dir: str):
    images_dir = Path(output_dir) / "images" / "train"
    labels_dir = Path(output_dir) / "labels" / "train"
    images_dir,mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok-True)

    candidates = [
        os.path.join(lisa_dir, "Annotations", "Annotations,csv"),
        os.path.join(lisa_dir, "frameAnnotationsBOX.csv"),
        os.path.join(lisa_dir, "allAnnotations.csv"),
    ]
    annotations_file = next((c for c in candidates if os.path.exists(c)), None)

    if not annotations_file:
        print(f" No Annotations file found")
        return 
    count = 0
    images_labels = {}

    with opne(annotations_file, "r") as f:
        reader = csv.reader(f, delimiter;";")
        next(reader, None)

        for row in reader:
            if len(row) < 6:
                continue
            img_path = row[0].strip()
            try:
                x1, y1, x2, y2 = float(row[1]), float(row[2]), float(row[3]), float(row[4])
            except (ValueError, IndexError):
                continue
            img_full = os.path.join(lisa_dir, img_path)
            if not os.path.exists(img_full):
                continue
            basename = os.path.basename(img_path)
            if basename not in image_labels:
                image_labels[basename] = {"src": img_full, "boxes": []}
            image_labels[basename]["boxes"].append((x1, y1, x2, y2))
    import cv2
    for basename, info in image_labels.items():
        dst_img = images_dir / basename
        if not dst_img.exists():
            shutil.copy2(info["src"], dst_img)
        img = cv2.imread(str(dst_img))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        label_name = Path(basename).stem + ".txt"
        with open(labels_dir / label_name, "w") as lf:
            for (x1, y1, x2, y2) in info["boxes"]:
                cx = max(0.0, min(1.0, ((x1 + x2) / 2.0) / iw))
                cy = max(0.0, min(1.0, ((y1 + y2) / 2.0) / ih))
                bw = max(0.0, min(1.0, (x2 - x1) / iw))
                bh = max(0.0, min(1.0, (y2 - y1) / ih))
                lf.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                count += 1
    print(f"  Converted {count} bounding boxes from {len(image_labels)} images.")
    return count

def train_yolo26_detector(data_yaml: str, epochs: int, imgsz: int = 320, batch: int = 16, base_model: str = "yolo26n.pt"):
    """Fine-tune the YOLO26-Nano detector on traffic light fixtures."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Error: ultralytics package required: pip install ultralytics")
    print(f"\n[TRAIN] Training YOLO26 Traffic Light Detector")
    print(f"  Base Model:  {base_model}")
    print(f"  Dataset:     {data_yaml}")
    print(f"  Resolution:  {imgsz}×{imgsz}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch:       {batch}\n")
    model = YOLO(base_model)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="light_detector_yolo26",
        project="runs/light_det",
        exist_ok=True,
        verbose=True,
    )
    best_path = os.path.join("runs", "light_det", "light_detector_yolo26", "weights", "best.pt")
    if os.path.exists(best_path):
        print(f"\n Training complete. Best checkpoint: {best_path}")
        return best_path
    return None



