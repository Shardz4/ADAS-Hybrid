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
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        os.path.join(lisa_dir, "Annotations", "Annotations.csv"),
        os.path.join(lisa_dir, "frameAnnotationsBOX.csv"),
        os.path.join(lisa_dir, "allAnnotations.csv"),
    ]
    annotations_file = next((c for c in candidates if os.path.exists(c)), None)

    if not annotations_file:
        print(f" No annotation file found in {lisa_dir}")
        return 0

    count = 0
    image_labels = {}

    with open(annotations_file, "r") as f:
        reader = csv.reader(f, delimiter=";")
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
    print(f"  Resolution:  {imgsz}x{imgsz}")
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

def export_to_onnx(weights_path: str, output: str, imgsz: int = 320):
    """Export the trained fixture detector to ONNX format."""
    from ultralytics import YOLO
    print(f"\n[EXPORT] Converting {weights_path} -> {output} (imgsz={imgsz})...")
    model = YOLO(weights_path)
    export_path = model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    if os.path.abspath(export_path) != os.path.abspath(output):
        if os.path.exists(output):
            os.remove(output)
        os.replace(export_path, output)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        dummy = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
        out = sess.run(None, {inp.name: dummy})
        print(f" Verification: Input '{inp.name}' {inp.shape} -> Output {out[0].shape}")
        print(f" YOLO26 fixture detector ONNX ready: {output}")
    except Exception as e:
        print(f"Verification warning: {e}")

def main():
    parser = argparse.ArgumentParser(description="Train and Export YOLO26 Traffic Light Fixture Detector")
    parser.add_argument("--base-model", type=str, default="yolo26n.pt",
                        help="Base YOLO26 model checkpoint (default: yolo26n.pt)")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset.yaml")
    parser.add_argument("--lisa-dir", type=str, default=None, help="Path to raw LISA dataset folder")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--export", action="store_true", help="Auto-export to ONNX upon completion")
    parser.add_argument("--output", type=str, default="models/light_det.onnx")
    args = parser.parse_args()
    if args.lisa_dir:
        yolo_dir = os.path.join("datasets", "lisa_yolo")
        convert_lisa_to_yolo(args.lisa_dir, yolo_dir)
        args.data = create_dataset_yaml(yolo_dir, os.path.join(yolo_dir, "dataset.yaml"))
    if not args.data:
        sys.exit("Error: Must provide --data <dataset.yaml> or --lisa-dir <folder>")
    best_ckpt = train_yolo26_detector(args.data, args.epochs, args.imgsz, args.batch, args.base_model)
    if args.export and best_ckpt:
        export_to_onnx(best_ckpt, args.output, args.imgsz)

if __name__ == "__main__":
    main()
