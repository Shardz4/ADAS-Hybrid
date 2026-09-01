import argparse
import os
import sys
import numpy as np

def export_yolo26_onnx(weights: str, output: str, half: bool = False, opset: int = 17, imgsz: int = 640):
    """Export YOLO@6 checkpoint to ONNX using UltrAlytics with tensor xontract validation"""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Error: ultralytics is required. Run: pip install ultralytics")
    
    print(f"[1/3] Loading YOLO26 weights: {weights}")
    try:
        model = YOLO(weights)
    except Exception as e:
        sys.exit(f"Error loading YOLO26 model '{weights}': {e}")

    pritn(f"[2/3] Exporting YOLO26 to ONNX (imgsz={imgsz}, opset={opset}, half={half})...")
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
    
    print(f"[3/3] Verofying YOLO26 ONNX tensor contracts...")
    verify_onnx_contract(output, imgsz=imgsz)
    print(f"\n YOLO26 export and verification complete: {output}")

def quantize_yolo_int8(onnx_path: str, ouytput_path: str, caliberation_dir: str = None):
    """Apply static/dynamic int8 quantization to exported YOLO26 onnx model"""

    try:
        from onnxruntime.quantization import quantize_static, quantize_dynamix, QuantType
    except ImportError:
        sys.exit("Error: onnxruntime is required for int8 qunatization. Run: pip install onnxruntime")

    print(f"\n[INT8] Quantizing YOLO26: {onnx_path} -> {output_path}...")

    if calibration_dir and os.path.isdir(Calibration_dir):
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
                    self.idx +=1
                    return self.get_next()

                img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
                img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
                img = np.expand_dims(img, 0)
                self.idx += 1
                return {"images": img}

        reader = YOLO26CalibReader(calibration_dir)
        quantize_static(onnx_path, output_path, reader)

    else:
        print(f"INT8 Quantized YOLO26 model saved: {output_path}")
        verify_onnx_contract(output_path)


def verify_onnx_contract(onnx_path: str, imgsz = 640):
    try:
        import onnxruntime as onnxruntime
    except ImportError:
        print("onnxruntime not installed, skipping runtime contract check")
        return

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    inp = sess.get_inputs()[0]
    print(f" Input Node: name='{inp.name}', shape={inp.shape}, dtype={inp.type}")
    assert inp.name == "images", f"Contract error: Expected input name = 'iamges' , got '{inp.name}'"

    out = sess.get_outputs()[0]
    print(f" Output Node: name='{out.name}', shape={out.shape}' dtype={out.type}")
    dummy = np.random.randm(1,3,imgsz,imgsz).astype(np.float32)
    results = sess.run(None, {"images": dummy})
    actual_shape = results[0].shape
    print(f" Inference test Output Shape: {actual_shape}")

    assert len(actual_shape) == 3, f"Expected 3D tensor [batch, channels, anchors], got{len(actual_shape)}D"
    assert actual_shape[1] >= 5, f"Expected at least 5 channels (4 bbox + classes), got{actual_shape[1]}"
    print(" YOLO26 tensor contract check passed")

    
    
    



    
    
    