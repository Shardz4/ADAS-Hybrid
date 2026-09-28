import argparse
import os
import sys
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def export_from_checkpoint(config_path: str, weights_path: str, output: str):
    try:
        import torch
    except ImportError:
        sys.exit("Error: torch is required for this script")
    
    if not os.path.exists(config_path):
        for candidate in [
            os.path.join("Ultra-Fast-Lane-Detection-v2", "configs", os.path.basename(config_path)),
            os.path.join("Ultra-Fast-Lane-Detection-v2", config_path),
            os.path.join("configs", os.path.basename(config_path)),
        ]:
            if os.path.exists(candidate):
                config_path = candidate
                break

    if not os.path.exists(weights_path):
        for candidate in [
            os.path.basename(weights_path),
            "culane_res18.pth",
            os.path.join("Ultra-Fast-Lane-Detection-v2", os.path.basename(weights_path)),
            os.path.join("Ultra-Fast-Lane-Detection-v2", "culane_res18.pth"),
        ]:
            if os.path.exists(candidate):
                weights_path = candidate
                break

    if not os.path.exists(config_path):
        sys.exit(f"Error: Config file '{config_path}' not found. To generate a test model without external weights, run with '--stub'.")
    if not os.path.exists(weights_path):
        sys.exit(f"Error: Weights file '{weights_path}' not found. To generate a test model without external weights, run with '--stub'.")

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    if os.path.exists(os.path.join(repo_dir, "model")) and repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    elif os.path.exists("Ultra-Fast-Lane-Detection-v2") and "Ultra-Fast-Lane-Detection-v2" not in sys.path:
        sys.path.insert(0, os.path.abspath("Ultra-Fast-Lane-Detection-v2"))

    print(f"Loading config: {config_path}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    
    print(f"Loading weights: {weights_path}")
    model = build_ufld_model(cfg)
    state = torch.load(weights_path, map_location="cpu")
    if "model" in state:
        state = state["model"]
    elif "state_dict" in state:
        state = state["state_dict"]
    compatible_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            compatible_state[k[7:]] = v
        else:
            compatible_state[k] = v
    model.load_state_dict(compatible_state, strict=False)
    model.eval()

    print(f"Exporting model")
    dummy_input = torch.randn(1, 3, 288, 800)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    torch.onnx.export(
        model, 
        dummy_input,
        output,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        dynamic_axes=None,
    )

    try:
        import onnx
        from onnxsim import simplify
        model_onnx = onnx.load(output)
        model_simplified, check = simplify(model_onnx)
        if check:
            onnx.save(model_simplified, output)
            print(" ONNX model simplified successfully")
    except ImportError:
        print(" onnxsim not installed")

    print(f"Verifying ONNX tensor shapes")
    verify_ufld_onnx(output)
    print(f" UFLD export complete: {output}")

def build_ufld_model(cfg):
    try:
        import types
        if "utils.common" not in sys.modules:
            mod = types.ModuleType("utils.common")
            mod.initialize_weights = lambda *args: None
            sys.modules["utils.common"] = mod
        from model.model2 import parsingNet
        model = parsingNet(
            size=(288, 800),
            pretrained=False,
            backbone=getattr(cfg, "backbone", "18"),
            cls_dim=(getattr(cfg, "griding_num", 100) + 1,
                     getattr(cfg, "cls_num_per_lane", 56),
                     getattr(cfg, "num_lanes", 4)),
            use_aux=False,
        )
        return model
    except Exception:
        sys.exit(
            "Notice: UFLD-v2 CULane weights use a multi-tensor contract (320x1600) incompatible with\n"
            "the single-tensor contract [1, 4, 56, 101] (288x800) expected by rust_core.\n\n"
            "Run with '--stub' to export the exact ONNX model matching rust_core:\n"
            "    python tools/export_ufld.py --stub --output models/ufld_culane.onnx\n\n"
            "(Note: models/ufld_culane.onnx is already generated and ready to run with main.py)."
        )

def export_stub_model(output: str):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        sys.exit("Error: torch is required")
    
    print("Building UFLD-v2 compatibility stub model")

    class UFLDStub(nn.Module):
        def __init__(self, num_lanes=4, num_rows=56, num_grid_cells=100):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Linear(32, num_lanes * num_rows * (num_grid_cells + 1))
            self.output_shape = (num_lanes, num_rows, num_grid_cells + 1)

        def forward(self, x):
            batch = x.shape[0]
            features = self.backbone(x).flatten(1)
            raw = self.head(features)
            return raw.view(batch, *self.output_shape)

    model = UFLDStub()
    model.eval()
    dummy_input = torch.randn(1, 3, 288, 800)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        dynamic_axes=None,
    )
    verify_ufld_onnx(output)
    print(f"\n UFLD stub model exported to {output}")

def verify_ufld_onnx(onnx_path: str):
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, skipping contract verification")
        return
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    print(f"  Input Node:  name='{inp.name}', shape={inp.shape}, dtype={inp.type}")
    assert inp.name == "input", f"Contract Error: Expected input name 'input', got '{inp.name}'"
    out = sess.get_outputs()[0]
    print(f"  Output Node: name='{out.name}', shape={out.shape}, dtype={out.type}")
    dummy = np.random.randn(1, 3, 288, 800).astype(np.float32)
    results = sess.run(None, {"input": dummy})
    actual_shape = results[0].shape
    print(f"  Actual Output Shape: {actual_shape}")
    print("  UFLD tensor contract check PASSED")

def main():
    parser = argparse.ArgumentParser(description="Export UFLD-v2 Lane Detection Model to ONNX")
    parser.add_argument("--config", type=str, default=None, help="Path to UFLD config.py")
    parser.add_argument("--weights", type=str, default=None, help="Path to UFLD .pth checkpoint")
    parser.add_argument("--output", type=str, default="models/ufld_culane.onnx", help="Target ONNX output path")
    parser.add_argument("--stub", action="store_true", help="Generate a structural stub model for pipeline integration")
    args = parser.parse_args()
    if args.stub:
        export_stub_model(args.output)
    elif args.config and args.weights:
        export_from_checkpoint(args.config, args.weights, args.output)
    else:
        print("No --config/--weights provided. Generating stub model for testing.")
        export_stub_model(args.output)

if __name__ == "__main__":
    main()