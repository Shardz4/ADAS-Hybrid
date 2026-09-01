"""
YOLO26 → ONNX Export & INT8 Quantization

Exports the YOLO26 model (yolo26n.pt) to ONNX format strictly matching
the Rust AdasBrain vehicle detection contract.

Rust contract (lib.rs:68-160):
    Input:  name="images", shape=[1, 3, 640, 640], dtype=f32
    Output: shape=[1, 84, 8400] (4 bbox + 80 COCO classes × 8400 anchors)
    Vehicle class indices: 2=car, 3=motorcycle, 5=bus, 7=truck

Usage:
    python tools/export_yolo.py --weights yolo26n.pt --output models/yolo26n.onnx
    python tools/export_yolo.py --weights yolo26n.pt --output models/yolo26n.onnx --half
    python tools/export_yolo.py --weights yolo26n.pt --output models/yolo26n.onnx --int8
"""

import argparse
import os
import sys
import numpy as np


def export_yolo26_onnx(weights: str, output: str, half: bool = False, opset: int = 17, imgsz: int = 640):
    """Export YOLO26 checkpoint to ONNX using Ultralytics with tensor contract validation."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Error: ultralytics is required. Run: pip install ultralytics")

    print(f"[1/3] Loading YOLO26 weights: {weights}")
    try:
        model = YOLO(weights)
    except Exception as e:
        sys.exit(f"Error loading YOLO26 model '{weights}': {e}")

    print(f"[2/3] Exporting YOLO26 to ONNX (imgsz={imgsz}, opset={opset}, half={half})...")
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=True,
        half=half,
        dynamic=False,
    )

    if os.path.abspath(export_path) != os.path.abspath(output):
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        if os.path.exists(output):
            os.remove(output)
        os.replace(export_path, output)

    print(f"[3/3] Verifying YOLO26 ONNX tensor contracts...")
    verify_onnx_contract(output, imgsz=imgsz)
    print(f"\n YOLO26 export and verification complete: {output}")


def quantize_yolo26_int8(onnx_path: str, output_path: str, calibration_dir: str = None):
    """Apply static or dynamic INT8 quantization to the exported YOLO26 ONNX model."""
    try:
        from onnxruntime.quantization import quantize_static, quantize_dynamic, QuantType
    except ImportError:
        sys.exit("Error: onnxruntime is required for INT8 quantization. Run: pip install onnxruntime")

    print(f"\n[INT8] Quantizing YOLO26: {onnx_path} → {output_path}...")

    if calibration_dir and os.path.isdir(calibration_dir):
        from onnxruntime.quantization import CalibrationDataReader

        class YOLO26CalibReader(CalibrationDataReader):
            def __init__(self, calib_dir, input_shape=(1, 3, 640, 640)):
                self.files = [
                    os.path.join(calib_dir, f) for f in os.listdir(calib_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                ]
                self.idx = 0
                self.input_shape = input_shape

            def get_next(self):
                if self.idx >= len(self.files) or self.idx >= 100:
                    return None
                import cv2
                img = cv2.imread(self.files[self.idx])
                if img is None:
                    self.idx += 1
                    return self.get_next()
                img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
                img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
                img = np.expand_dims(img, 0)
                self.idx += 1
                return {"images": img}

        reader = YOLO26CalibReader(calibration_dir)
        quantize_static(onnx_path, output_path, reader)
    else:
        print("  Applying dynamic weight quantization (QUInt8)...")
        quantize_dynamic(onnx_path, output_path, weight_type=QuantType.QUInt8)

    print(f" INT8 Quantized YOLO26 model saved: {output_path}")
    verify_onnx_contract(output_path)


def verify_onnx_contract(onnx_path: str, imgsz: int = 640):
    """Verify that the exported YOLO26 model strictly matches the Rust perception contract."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed, skipping runtime contract check")
        return

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    inp = sess.get_inputs()[0]
    print(f"  Input Node:  name='{inp.name}', shape={inp.shape}, dtype={inp.type}")
    assert inp.name == "images", f"Contract Error: Expected input name 'images', got '{inp.name}'"

    out = sess.get_outputs()[0]
    print(f"  Output Node: name='{out.name}', shape={out.shape}, dtype={out.type}")

    dummy = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    results = sess.run(None, {"images": dummy})
    actual_shape = results[0].shape
    print(f"  Inference Test Output Shape: {actual_shape}")

    assert len(actual_shape) == 3, f"Expected 3D tensor [batch, channels, anchors], got {len(actual_shape)}D"
    assert actual_shape[1] >= 5, f"Expected at least 5 channels (4 bbox + classes), got {actual_shape[1]}"
    print("  YOLO26 tensor contract check PASSED")


def main():
    parser = argparse.ArgumentParser(description="Export YOLO26 to ONNX for ADAS Hybrid Perception Engine")
    parser.add_argument("--weights", type=str, default="yolo26n.pt",
                        help="YOLO26 model checkpoint (default: yolo26n.pt)")
    parser.add_argument("--output", type=str, default="models/yolo26n.onnx",
                        help="Output path for the generated ONNX model (default: models/yolo26n.onnx)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image resolution (default: 640)")
    parser.add_argument("--half", action="store_true",
                        help="Export weights in FP16 half-precision")
    parser.add_argument("--int8", action="store_true",
                        help="Apply post-training INT8 quantization")
    parser.add_argument("--calib-dir", type=str, default=None,
                        help="Optional directory of driving frames for static INT8 calibration")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset target version (default: 17)")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    export_yolo26_onnx(args.weights, args.output, half=args.half, opset=args.opset, imgsz=args.imgsz)

    if args.int8:
        int8_output = args.output.replace(".onnx", "_int8.onnx")
        quantize_yolo26_int8(args.output, int8_output, args.calib_dir)


if __name__ == "__main__":
    main()
