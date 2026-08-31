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
    
    