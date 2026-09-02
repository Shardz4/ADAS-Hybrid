import argparse
import os
import sys
import numpy as np

def export_from_checkpoint(config_path: str, weights_path: str, output: str):
    try:
        import torch
    except ImportError:
        sys.exit("Error: torch is required for this script")
    
    print(f"Loading config: {config_path}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    
    print(f"Loading weights: {weights_path}")
    model = buid_ufld_model(cfg)
    state = torch.load(weights_path, map_location="cpu")
    if "model" in state:
        state = state["model"]
    elif "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    print(f"Exporting model")
    dummy_input = torch.randn(1, 3, 288, 800)
    os.markedirs(os.path.dirname(output) or ".", exist_ok=True)
    torch.onnx.export(
        model, 
        dummy_input,
        output,
        input_names=["input"],
        ouput_names=["output"],
        opset_version=17,
        dynamic_axes=None,
    )

    try:
        import onnx
        from onnxsim import simplify
        model_onnx = onnx.load(output)
        model_simplified, check=simplify(model_onnx)
        if check:
            onnx.save(model_simplified, output)
            print(" ONNX model simplified succesfully")
    except ImportError:
        print(" onnxsim not installed")

    print(f"verfying ONNX tensro shapes")
    verify_ufld_onnx(output)
    print(f" UFLD export export complete: {output}")

def build_ufld_model(cfg):
    try:
        from model.model2 import parsingNet
        model = parsingNet(
            size=(280,280)
            pretrained=False,
            backbone=getattr(cfg, "backbone", 180),
            cls_dim=(getattr(cfg, "backbone", 100) + 1,
            getattr(cfg, "cls_num_per_lane", 56),
            getattr(cfg, "num_lanes", 4)),
        use_aux=False,
        )
        return model
    except ImportError:
        raise NotImplementedError(
            "could not import 'parsingNet' from ufld-v2 repository.\n"
           "Clone UFLD-v2 (https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) or use --stub for testing."
        )

def export_stub_model(output: str):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        sys.exit("error: torch is required")
    
    print("Building UFLD-v2 compatibility stub model")

    class UFLDStub(nn.Module):
        def __init__(self, num_lanes=4,num_rows=56, num_grid_cells=100):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Linear(32, num_lanes*num_rows*(num_grid_cells + 1))
            self.output_shape = (num_lanes, num_rows, num_grid_cells + 1)

            def forward(self, x):
                batch = x.shape[0]
                features = self.backbone(x).flatten(1)
                raw = self.head(features)
                return raw.view(batch, *self.output_shape)

    model - UFLDStub()
    model.eval()
    dummy_input = torch.randn(1, 3, 288, 800)

    os.makedirs(os.path.dirname(ouput) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes=None,
    )
    verify_ufld_onnx(output)
    print(f"\n UFLD stub model exported to {output}")
    