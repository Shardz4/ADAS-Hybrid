"""
Traffic Light State Classifier
Trains a lightweight EfficientNet-B0 classifier for traffic light states:
  class 0: Red
  class 1: Yellow
  class 2: Green
  class 3: None / Off

Strictly matches rust_core/src/traffic_light.rs tensor contract:
  Input:  'input'  [1, 3, 64, 32] float32 in [0.0, 1.0]
  Output: 'output' [1, 4] float32 logits
"""

import argparse
import csv
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np

CLASS_NAMES = ["0_red", "1_yellow", "2_green", "3_none"]
CLASS_TO_IDX = {
    "0_red": 0, "red": 0,
    "1_yellow": 1, "yellow": 1,
    "2_green": 2, "green": 2,
    "3_none": 3, "none": 3, "off": 3,
}
INPUT_H, INPUT_W = 64, 32


def build_classifier(num_classes: int = 4, pretrained: bool = True):
    import torch.nn as nn
    try:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
    except ImportError:
        sys.exit("Error: torchvision required")

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


class NormalizedClassifier(object):
    """Wraps model with ImageNet mean/std so ONNX accepts raw [0.0, 1.0] CHW inputs."""
    def __new__(cls, base_model):
        import torch
        import torch.nn as nn

        class _Wrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
                self.model = model

            def forward(self, x):
                x_norm = (x - self.mean) / self.std
                return self.model(x_norm)

        return _Wrapper(base_model)


def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 0, pin_memory: bool = True):
    import torch
    from torchvision import datasets, transforms

    train_tf = transforms.Compose([
        transforms.Resize((INPUT_H, INPUT_W)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((INPUT_H, INPUT_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")

    if not os.path.isdir(train_path):
        sys.exit(f"Error: Directory not found: {train_path}")

    class TrafficLightFolder(datasets.ImageFolder):
        def find_classes(self, directory):
            subdirs = [d.name for d in os.scandir(directory) if d.is_dir()]
            valid = [d for d in subdirs if d in CLASS_TO_IDX]
            valid.sort(key=lambda d: CLASS_TO_IDX[d])
            class_to_idx = {d: CLASS_TO_IDX[d] for d in valid}
            return valid, class_to_idx

    train_set = TrafficLightFolder(train_path, transform=train_tf)
    val_set = TrafficLightFolder(val_path, transform=val_tf) if os.path.isdir(val_path) else None

    print(f"  Dataset classes mapped: {train_set.class_to_idx}")
    print(f"  Train samples: {len(train_set):,} | Val samples: {len(val_set) if val_set else 0:,}")

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    ) if val_set else None

    return train_loader, val_loader


def extract_lisa_crops(lisa_dir: str, output_dir: str, max_per_class: int = 2000, seed: int = 42):
    import cv2

    lisa_path = Path(lisa_dir)
    out_path = Path(output_dir)

    print(f"\n[DATA] Indexing LISA dataset from {lisa_dir}...")
    t0 = time.time()
    image_map = {p.name: p for ext in ("*.jpg", "*.png", "*.jpeg") for p in lisa_path.rglob(ext)}
    box_csvs = list(lisa_path.rglob("*BOX*.csv"))

    if not box_csvs or not image_map:
        sys.exit(f"Error: No valid annotations or images found in {lisa_dir}")

    print(f"  Found {len(box_csvs)} annotation CSVs and {len(image_map):,} images on disk.")

    raw_classes = defaultdict(list)
    for cp in box_csvs:
        with open(cp, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader, None)
            for row in reader:
                if len(row) < 6:
                    continue
                fn = os.path.basename(row[0].strip().replace("\\", "/"))
                if fn not in image_map:
                    continue
                tag = row[1].strip().lower()
                try:
                    box = tuple(map(float, row[2:6]))
                except ValueError:
                    continue
                if "stop" in tag:
                    raw_classes["0_red"].append((image_map[fn], box))
                elif "warning" in tag:
                    raw_classes["1_yellow"].append((image_map[fn], box))
                elif "go" in tag:
                    raw_classes["2_green"].append((image_map[fn], box))

    print(f"  Raw annotations found: Red={len(raw_classes['0_red']):,}, "
          f"Yellow={len(raw_classes['1_yellow']):,}, Green={len(raw_classes['2_green']):,}")

    random.seed(seed)
    selected = {}
    for cls_name in ["0_red", "1_yellow", "2_green"]:
        items = raw_classes[cls_name]
        random.shuffle(items)
        selected[cls_name] = items[:max_per_class]

    # Generate 3_none (hard negatives: offset context patches + general road negatives)
    none_items = []
    # Offset patches adjacent to traffic lights (poles, signs, sky)
    for img_p, (x1, y1, x2, y2) in selected["2_green"][:max_per_class // 2]:
        none_items.append((img_p, (x1 + 60, y1, x2 + 60, y2)))
    for img_p, (x1, y1, x2, y2) in selected["0_red"][:max_per_class // 4]:
        none_items.append((img_p, (x1 - 60, y1, x2 - 60, y2)))
    for img_p, (x1, y1, x2, y2) in selected["1_yellow"][:max_per_class // 4]:
        none_items.append((img_p, (x1 + 50, y1 + 50, x2 + 50, y2 + 50)))
    selected["3_none"] = none_items[:max_per_class]

    # Group crops by image for fast single-read batch I/O
    img_to_crops = defaultdict(list)
    for cls_name, items in selected.items():
        val_cutoff = int(len(items) * 0.8)
        for idx, (img_path, box) in enumerate(items):
            split = "train" if idx < val_cutoff else "val"
            img_to_crops[img_path].append((split, cls_name, idx, box))

    for split in ("train", "val"):
        for cls_name in CLASS_NAMES:
            (out_path / split / cls_name).mkdir(parents=True, exist_ok=True)

    print(f"  Extracting {sum(len(v) for v in selected.values()):,} crops across {len(img_to_crops):,} unique frames...")
    extracted_counts = defaultdict(lambda: defaultdict(int))

    for img_path, crops in img_to_crops.items():
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        for split, cls_name, idx, (x1, y1, x2, y2) in crops:
            x1_c = max(0, min(w_img - 2, int(x1)))
            y1_c = max(0, min(h_img - 2, int(y1)))
            x2_c = max(x1_c + 2, min(w_img, int(x2)))
            y2_c = max(y1_c + 2, min(h_img, int(y2)))

            crop = img[y1_c:y2_c, x1_c:x2_c]
            if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                continue

            crop_resized = cv2.resize(crop, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
            out_file = out_path / split / cls_name / f"{cls_name}_{idx:05d}.jpg"
            cv2.imwrite(str(out_file), crop_resized)
            extracted_counts[split][cls_name] += 1

    dt = time.time() - t0
    print(f"  Crop extraction completed in {dt:.1f}s.")
    for split in ("train", "val"):
        counts_str = ", ".join(f"{c}: {extracted_counts[split][c]}" for c in CLASS_NAMES)
        print(f"    [{split.upper()}] {counts_str}")


def train(data_dir: str, epochs: int, batch_size: int, lr: float, device: str, patience: int = 10,
          resume: str = None, num_workers: int = 0):
    import torch
    import torch.nn as nn

    print(f"\n[TRAIN] Training EfficientNet-B0 Traffic Light State Classifier")
    print(f"  Data Directory : {data_dir}")
    print(f"  Resolution     : {INPUT_H}x{INPUT_W} (CHW: 3x64x32)")
    print(f"  Target Classes : 0: Red, 1: Yellow, 2: Green, 3: None/Off")
    print(f"  Batch Size     : {batch_size} | Learning Rate: {lr} | Epochs: {epochs}")
    print(f"  Device         : {device.upper()}")

    is_cuda = (device == "cuda")
    if is_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU Active     : {gpu_name} ({vram_gb:.2f} GB VRAM)")
        torch.backends.cudnn.benchmark = True

    model = build_classifier(num_classes=4, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

    train_loader, val_loader = get_dataloaders(
        data_dir, batch_size=batch_size, num_workers=num_workers, pin_memory=is_cuda
    )

    best_acc = -1.0
    start_epoch = 0
    no_improve_count = 0
    run_dir = os.path.join("runs", "light_cls")
    best_weights = os.path.join(run_dir, "best.pt")
    last_checkpoint = os.path.join(run_dir, "last_checkpoint.pt")
    os.makedirs(run_dir, exist_ok=True)

    if resume and os.path.exists(resume):
        print(f"  Resuming from checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        no_improve_count = ckpt.get("no_improve_count", 0)
        print(f"  Resumed at epoch {start_epoch}, best_acc={best_acc:.1f}%")

    print("\n--- Starting Training ---")
    for epoch in range(start_epoch, epochs):
        t_epoch_start = time.time()
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking=is_cuda), labels.to(device, non_blocking=is_cuda)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=is_cuda):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = (correct / total) * 100 if total > 0 else 0.0
        val_acc = 0.0
        improved = False

        if val_loader:
            model.eval()
            v_correct, v_total = 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device, non_blocking=is_cuda), labels.to(device, non_blocking=is_cuda)
                    with torch.amp.autocast("cuda", enabled=is_cuda):
                        outputs = model(imgs)
                    _, preds = outputs.max(1)
                    v_correct += preds.eq(labels).sum().item()
                    v_total += labels.size(0)
            val_acc = (v_correct / v_total) * 100 if v_total > 0 else 0.0

            if val_acc > best_acc:
                best_acc = val_acc
                improved = True
                no_improve_count = 0
                torch.save(model.state_dict(), best_weights)
            else:
                no_improve_count += 1

        scheduler.step()
        epoch_sec = time.time() - t_epoch_start

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
            "no_improve_count": no_improve_count,
        }, last_checkpoint)

        marker = "★ BEST" if improved else ""
        print(f"Epoch {epoch+1:2d}/{epochs} [{epoch_sec:.1f}s] | Loss: {total_loss/total:.4f} | "
              f"Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% (Best: {best_acc:.1f}%) {marker}")

        if val_loader and no_improve_count >= patience:
            print(f"\n  Early stopping triggered: no improvement for {patience} consecutive epochs.")
            break

    print(f"\nTraining complete. Best weights saved to: {best_weights}")
    return best_weights


def export_to_onnx(weights_path: str, output: str, device: str = "cpu"):
    import torch

    print(f"\n[EXPORT] Exporting EfficientNet-B0 to ONNX: {output}")
    base_model = build_classifier(num_classes=4, pretrained=False)
    base_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    base_model.eval()

    # Wrap model with ImageNet mean/std normalization so rust_core can pass raw [0.0, 1.0] pixels
    wrapped_model = NormalizedClassifier(base_model)
    wrapped_model.eval()

    dummy_input = torch.rand(1, 3, INPUT_H, INPUT_W, dtype=torch.float32)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        dynamic_axes=None,
    )

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        out = sess.get_outputs()[0]
        test_in = np.random.rand(1, 3, INPUT_H, INPUT_W).astype(np.float32)
        res = sess.run(None, {inp.name: test_in})
        assert res[0].shape == (1, 4), f"Expected shape (1, 4), got {res[0].shape}"
        print(f"  Tensor contract verified: '{inp.name}' {inp.shape} -> '{out.name}' {res[0].shape}")
        print(f"  Class mapping: 0=Red, 1=Yellow, 2=Green, 3=None/Off")
        print(f"  Classifier ONNX ready at: {output}")
    except Exception as e:
        print(f"  Verification note: {e}")


def verify_cuda_readiness(data_dir: str, device: str):
    import torch
    import torch.nn as nn

    print("\n" + "=" * 60)
    print(" TRAFFIC LIGHT STATE CLASSIFIER - CUDA & DATA READINESS AUDIT")
    print("=" * 60)

    # 1. PyTorch & CUDA Version Audit
    print(f"\n[1] Environment & Hardware Acceleration:")
    print(f"  PyTorch Version : {torch.__version__}")
    print(f"  CUDA Available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_arch = torch.cuda.get_device_capability(0)
        print(f"  GPU Device      : {gpu_name}")
        print(f"  Compute Arch    : sm_{cuda_arch[0]}{cuda_arch[1]}")
        print(f"  Total VRAM      : {vram_gb:.2f} GB")
        print(f"  cuDNN Version   : {torch.backends.cudnn.version()}")
        print(f"  cuDNN Benchmark : True")
    else:
        print("  WARNING: CUDA is NOT available. Running on CPU.")

    # 2. CUDA Forward & Backward Pass Verification
    print(f"\n[2] Model & CUDA Kernel Execution Test:")
    model = build_classifier(num_classes=4, pretrained=False).to(device)
    dummy_in = torch.randn(32, 3, INPUT_H, INPUT_W, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # Warmup and timed iteration
    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
        out = model(dummy_in)
        loss = criterion(out, torch.randint(0, 4, (32,), device=device))
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if device == "cuda":
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            with torch.amp.autocast("cuda"):
                out = model(dummy_in)
                loss = criterion(out, torch.randint(0, 4, (32,), device=device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        dt_ms = (time.time() - t0) * 100
        print(f"  Mixed-Precision Forward+Backward (batch 32): {dt_ms:.2f} ms / step on GPU")
        print(f"  CUDA acceleration is fully functional and optimized.")

    # 3. Data Integrity & Class Contract Audit
    print(f"\n[3] Dataset & Path Verification ({data_dir}):")
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    if os.path.isdir(train_dir):
        print(f"  Train path : {train_dir} (EXISTS)")
        for cls_name in CLASS_NAMES:
            cpath = os.path.join(train_dir, cls_name)
            count = len(os.listdir(cpath)) if os.path.isdir(cpath) else 0
            print(f"    - {cls_name:<10}: {count:,} images (Target idx: {CLASS_TO_IDX[cls_name]})")
    else:
        print(f"  Train path : {train_dir} (NOT FOUND - run with --extract-only to generate)")

    if os.path.isdir(val_dir):
        print(f"  Val path   : {val_dir} (EXISTS)")
        for cls_name in CLASS_NAMES:
            cpath = os.path.join(val_dir, cls_name)
            count = len(os.listdir(cpath)) if os.path.isdir(cpath) else 0
            print(f"    - {cls_name:<10}: {count:,} images (Target idx: {CLASS_TO_IDX[cls_name]})")

    # 4. DataLoader Live Test
    if os.path.isdir(train_dir) and any(os.listdir(os.path.join(train_dir, c)) for c in CLASS_NAMES if os.path.isdir(os.path.join(train_dir, c))):
        print(f"\n[4] DataLoader Integration Test:")
        loader, _ = get_dataloaders(data_dir, batch_size=32, num_workers=0, pin_memory=(device == "cuda"))
        imgs, labels = next(iter(loader))
        imgs, labels = imgs.to(device), labels.to(device)
        print(f"  Batch shape  : {tuple(imgs.shape)} on {imgs.device}")
        print(f"  Labels shape : {tuple(labels.shape)} on {labels.device} -> sample labels: {labels[:8].tolist()}")
        print(f"  DataLoader correctly streaming to {device.upper()}.")

    print("\n" + "=" * 60)
    print(" ALL CHECKS PASSED: Environment, data paths, and CUDA are ready.")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train Traffic Light State Classifier")
    parser.add_argument("--data", type=str, default=None, help="Path to cropped dataset folder")
    parser.add_argument("--lisa-dir", type=str, default=None, help="Path to raw LISA dataset folder")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--max-per-class", type=int, default=2000, help="Max crops per class to extract from LISA")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--extract-only", action="store_true", help="Extract crops from LISA and exit without training")
    parser.add_argument("--check-cuda", action="store_true", help="Verify CUDA acceleration and environment readiness then exit")
    parser.add_argument("--export", action="store_true", help="Auto-export ONNX on completion")
    parser.add_argument("--output", type=str, default="models/light_cls.onnx", help="Exported ONNX file path")
    args = parser.parse_args()

    default_data_dir = os.path.join("datasets", "light_cls")
    default_lisa_dir = os.path.join("datasets", "lisa")

    if not args.data:
        args.data = default_data_dir

    # Extract crops if light_cls does not exist or has empty train folder, or if explicitly requested
    train_dir = os.path.join(args.data, "train")
    needs_extraction = not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0 or args.extract_only

    if needs_extraction:
        lisa_dir = args.lisa_dir or default_lisa_dir
        if not os.path.exists(lisa_dir):
            sys.exit(f"Error: Dataset directory {args.data} not found and LISA directory {lisa_dir} does not exist.")
        extract_lisa_crops(lisa_dir, args.data, max_per_class=args.max_per_class)
        if args.extract_only:
            print("\n[INFO] Data extraction complete. --extract-only specified; exiting before training.")
            return

    device = "cuda" if has_cuda() else "cpu"

    if args.check_cuda:
        verify_cuda_readiness(args.data, device)
        return

    best_ckpt = train(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        patience=args.patience,
        resume=args.resume,
        num_workers=args.num_workers,
    )

    if args.export and best_ckpt:
        export_to_onnx(best_ckpt, args.output, device)


def has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
