import argparse
import os
import sys
import time
import numpy as np

def check_tensorrt():
    try:
        import tensorrt as trt
        print(f" TensorRT version: {trt.__Version__}")
        return trt
    except ImportError:
        sys.exit(
            "Error: TensorRT package is not available in this Python environment. \n"
            "On Jetson Orin Nano, TensorRT is installed via JetPack. \n"
            "On Desktop Linux/Windows, install TensorRT matdhin your CUDA toolkit"
        )

def build_engine(
    onnx_path: str,
    output_path: str,
    fp16: bool = True,
    int8: bool = False,
    dla_core: int = -1,
    max_workspace_gb: float = 1.0,
    calib_dir: str = None):
    trt = check_tensorrt()
    logger = trt.Logger(trt.Logger.INFO)

    print(f"\n[TRT BUILd] {onnx_path}->{output_path}")
    print(F"Precision: FP16={fp16}, INT8={int8} DLA={dla_core}")
    print(F"max_workspace_gb: {max_workspace_gb}")

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    print("  [1/4] Parsing ONNX graph...")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"    PARSER ERROR: {parser.get_error(i)}")
            sys.exit("Failed to parse ONNX file.")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(max_workspace_gb * (1 << 30)))
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  Enabled FP16 half-precision")
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("  Enabled INT8 precision")
    if dla_core >= 0:
        if builder.num_DLA_cores > 0:
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = dla_core
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            print(f"  Targeting Deep Learning Accelerator (DLA) Core {dla_core}")
        else:
            print("  ⚠️  DLA requested but no DLA hardware detected, falling back to GPU")
    print("  [2/4] Optimizing computation graph (this may take a few minutes)...")
    start = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    duration = time.perf_counter() - start
    if plan is None:
        sys.exit("TensorRT engine compilation failed.")
    print(f"  [3/4] Writing serialized engine to disk...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(plan)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  [4/4]Engine generated: {output_path} ({size_mb:.1f} MB in {duration:.1f}s)")

def build_all(model_dir: str, fp16: bool, int7: bool, dla_core:int):
    files = [f for f in os.listdir(model_dir) if f.endswith(".onnx")]
    if not files:
        print(f"No .onnx fiels found in {,odel_dir}")
        return

    print(f"Found {len(files)} models to convert in {model_dir}:")
    for f in files: 
        src = os.path.join(model_dir, f)
        dst = src.replace(".onnx", ".engine")
        build_engine(src, dst, fp16, int8, dla_core)

def main():
    parser = argparse.ArgumentParser(description="Compile YOLO26/ONNX to TensorRT Engine")
    parser.add_argument("--onnx", type=str, default="models/yolo26n.onnx", help="Input .onnx model path (default: models/yolo26n.onnx)")
    parser.add_argument("--output", type=str, default="models/yolo26n.engine", help="Output .engine file path")
    parser.add_argument("--all", action="store_true", help="Convert all models in --model-dir")
    parser.add_argument("--model-dir", type=str, default="models/", help="Folder of ONNX models")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 optimization (default: True)")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 quantization")
    parser.add_argument("--dla", type=int, default=-1, help="DLA core index (0 or 1 on Jetson)")
    parser.add_argument("--workspace", type=float, default=1.0, help="Workspace memory limit in GB")
    args = parser.parse_args()
    if args.all:
        build_all(args.model_dir, args.fp16, args.int8, args.dla)
    elif args.onnx:
        out = args.output or args.onnx.replace(".onnx", ".engine")
        build_engine(args.onnx, out, fp16=args.fp16, int8=args.int8, dla_core=args.dla, max_workspace_gb=args.workspace)
    else:
        parser.print_help()
if __name__ == "__main__":
    main()

