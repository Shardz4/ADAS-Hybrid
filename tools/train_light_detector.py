import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
import numpy as np


def create_dataset_yaml(data_dir: str, output_yaml: str):
    """Generate YOLO dataset YAML config for single-class traffic light detection."""
    abs_data = os.path.abspath(data_dir)
    train_dir = "train/images" if os.path.exists(os.path.join(abs_data, "train", "images")) else "images/train"
    val_dir = "val/images" if os.path.exists(os.path.join(abs_data, "val", "images")) else ("images/val" if os.path.exists(os.path.join(abs_data, "images", "val")) else train_dir)
    yaml_content = f"""# Traffic Light Detection Dataset
path: {abs_data}
train: {train_dir}
val: {val_dir}
nc: 1
names:
  0: traffic_light
"""
    with open(output_yaml, "w") as f:
        f.write(yaml_content.strip())
    print(f"  Dataset configuration created at: {output_yaml}")
    return output_yaml


def convert_lisa_to_yolo(lisa_dir: str, output_dir: str, max_images: int = None):
    train_img = Path(output_dir) / "train" / "images"
    train_lbl = Path(output_dir) / "train" / "labels"
    val_img = Path(output_dir) / "val" / "images"
    val_lbl = Path(output_dir) / "val" / "labels"
    train_img.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(parents=True, exist_ok=True)
    val_img.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)

    print("  Indexing LISA image files on disk...")
    image_map = {}
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        for p in Path(lisa_dir).rglob(ext):
            image_map[p.name] = p

    box_csvs = list(Path(lisa_dir).rglob("*BOX*.csv"))
    if not box_csvs:
        box_csvs = [p for p in Path(lisa_dir).rglob("*.csv") if "annotation" in p.name.lower()]

    if not box_csvs:
        print(f" No annotation files found in {lisa_dir}")
        return 0

    print(f"  Found {len(box_csvs)} annotation CSVs and {len(image_map):,} images.")
    image_labels = {}
    count = 0

    for csv_path in box_csvs:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader, None)
            for row in reader:
                if len(row) < 6:
                    continue
                basename = os.path.basename(row[0].strip().replace("\\", "/"))
                if basename not in image_map:
                    continue
                try:
                    x1, y1, x2, y2 = float(row[2]), float(row[3]), float(row[4]), float(row[5])
                except (ValueError, IndexError):
                    continue
                if basename not in image_labels:
                    image_labels[basename] = {"src": image_map[basename], "boxes": []}
                image_labels[basename]["boxes"].append((x1, y1, x2, y2))

    items = list(image_labels.items())
    if max_images and len(items) > max_images:
        import random
        random.seed(42)
        random.shuffle(items)
        items = items[:max_images]

    val_cutoff = int(len(items) * 0.8) if len(items) > 1 else len(items)
    iw, ih = 1280, 960
    for idx, (basename, info) in enumerate(items):
        target_img_dir = train_img if idx < val_cutoff else val_img
        target_lbl_dir = train_lbl if idx < val_cutoff else val_lbl

        dst_img = target_img_dir / basename
        if not dst_img.exists():
            try:
                os.link(info["src"], dst_img)
            except Exception:
                shutil.copy2(info["src"], dst_img)

        label_name = Path(basename).stem + ".txt"
        with open(target_lbl_dir / label_name, "w") as lf:
            for (x1, y1, x2, y2) in info["boxes"]:
                cx = max(0.0, min(1.0, ((x1 + x2) / 2.0) / iw))
                cy = max(0.0, min(1.0, ((y1 + y2) / 2.0) / ih))
                bw = max(0.0, min(1.0, (x2 - x1) / iw))
                bh = max(0.0, min(1.0, (y2 - y1) / ih))
                lf.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                count += 1
    print(f"  Converted {count} bounding boxes from {len(items)} images into {output_dir}.")
    return count

def train_yolo26_detector(data_yaml: str, epochs: int, imgsz: int = 320, batch: int = 16, base_model: str = "yolo26n.pt", patience: int = 10, resume: bool = False, device: str = None):
    """Fine-tune the YOLO26-Nano detector on traffic light fixtures."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Error: ultralytics package required: pip install ultralytics")
    import torch
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and str(device) != "cpu" else "CPU"
    print(f"\n[TRAIN] Training YOLO26 Traffic Light Detector")
    print(f"  Base Model:  {base_model}")
    print(f"  Dataset:     {data_yaml}")
    print(f"  Resolution:  {imgsz}x{imgsz}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch:       {batch}")
    print(f"  Device:      {device} ({dev_name})")
    print(f"  Patience:    {patience} (early stopping)")
    print(f"  Resume:      {resume}\n")
    model = YOLO(base_model)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        resume=resume,
        save=True,
        save_period=1,
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
    parser.add_argument("--max-images", type=int, default=None, help="Optional limit on images to convert for faster training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint after crash")
    parser.add_argument("--export", action="store_true", help="Auto-export to ONNX upon completion")
    parser.add_argument("--device", type=str, default=None, help="Device to train on (e.g. 0, cpu)")
    parser.add_argument("--output", type=str, default="models/light_det.onnx")
    args = parser.parse_args()

    if not args.data and not args.lisa_dir:
        if os.path.exists(os.path.join("datasets", "lisa")):
            args.lisa_dir = os.path.join("datasets", "lisa")
        elif os.path.exists(os.path.join("datasets", "lisa.yaml")):
            args.data = os.path.join("datasets", "lisa.yaml")

    if args.lisa_dir:
        if not os.path.exists(args.lisa_dir):
            sys.exit(f"Error: LISA dataset directory '{args.lisa_dir}' does not exist.")
        yolo_dir = os.path.join("datasets", "lisa_yolo")
        convert_lisa_to_yolo(args.lisa_dir, yolo_dir, max_images=args.max_images)
        args.data = create_dataset_yaml(yolo_dir, os.path.join(yolo_dir, "dataset.yaml"))

    if not args.data:
        sys.exit("Error: Must provide --data <dataset.yaml> or --lisa-dir <folder>")
    if not os.path.exists(args.data):
        sys.exit(f"Error: Dataset YAML '{args.data}' not found. Download the LISA dataset and use '--lisa-dir <folder>', or provide a valid YOLO dataset yaml.")
    best_ckpt = train_yolo26_detector(args.data, args.epochs, args.imgsz, args.batch, args.base_model, args.patience, args.resume, device=args.device)
    if args.export and best_ckpt:
        export_to_onnx(best_ckpt, args.output, args.imgsz)

if __name__ == "__main__":
    main()
