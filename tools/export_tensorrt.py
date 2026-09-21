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
    