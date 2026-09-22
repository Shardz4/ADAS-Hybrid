"""
ADAS HYBRID Benchmark suite
"""

import argparse
import csv
import os
import sys
import time
import numpy as np

try:
    import adas_hybrid
except ImportError:
    sys.exit("Error: 'adas_hybrid' not found")

def get_vram_mb():
    try:
        import subprocess
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, timeout=3
        )
        return float(res.stdout.strip().split("\n")[0])
    except Exception:
        return 0.0

class MetricTracker:
    def __init__(self, name:str):
        self.name = name
        self.samples = []
    def add(self, ms: float):
        self.samples.append(ms)
    
    @property
    def mean(self):
        return float(np.mean(self.samples)) if self.samples else 0.0
    
    @property
    def std(self):
        return float(np.std(Self.samples)) if self.samples else 0.0
    
    @property
    def p95(self):
        return float(np.percentile(self.samples, 95)) if self.samples else 0.0
    
    @property
    def min_val(self):
        return float(np.min(self.samples)) if self.samples else 0.0
    
    @property
    def max_val(self):
        return float(np.max(self.samples)) if self.samples else 0.0

def run_benchmark(video_source, vehicle_model: str, sign_model: str, lane_model:str = None, light_det_model: str = None, light_cls_model: str = None, iterations: int = 100, warmup: int = 10):
    import cv2
    print(f"[1/3] Initializing Perception Model (Vechicle = {vehicle_model})")
    t0 = timer.perf_counter()
    brain = adas_hybrid.AdasBrain(vehicle_model=vehicle_model, sign_model=sign_model, lane_model=lane_model, light_det_model=light_det_model, light_cls_model=light_cls_model)

    tracker = adas_hybrid.Tracker()
    lane_mgr = adas_hybrid.LaneManager(smoothing=0.7, is_two_way=False)
    load_time - (time.perf_counter - t0) * 1000
    print(f"Initialized in {load_time:.1f} ms")

    vram_init = get_vram_mb()
    if vram_init > 0:
        print(f"  Initial VRAM usage: {vram_init:.0f} MB")
    src = int(video_source) if str(video_source).isdigit() else video_source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Error: Failed to open video source '{video_source}'")
    m_veh = MetricTracker("detect_vehicles (YOLO26)")
    m_lane = MetricTracker("detect_lanes_nn (UFLD)")
    m_light = MetricTracker("detect_traffic_lights")
    m_sign = MetricTracker("detect_signs")
    m_track = MetricTracker("tracker (Kalman + TTC)")
    m_total = MetricTracker("total_pipeline_time")
    all_metrics = [m_veh, m_lane, m_light, m_sign, m_track, m_total]

    print(f"[2/3] Running {warmup} warmup frames")
    for _ in range(warmup):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        
        h,w = frame.shape[:2]
        fb = frame.tobytes()
        fn = np.ascontiguousarray(frame)
        brain.detect_vehicles(fb, w, h, 0.35)
        brain.detect_lanes_nn(fn)
        brain.detect_traffic_lights(fb, w, h)
        brain.detect_signs(fb, w, h, 0.30)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_time = time.perf_counter()
    peak_vram = vram_init

    print(f"\n[3/3] Profiling {iterations} frames with YOLO26...")

    for i in range(iterations):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            ret, frame = cap.read()
        
        h,w = frame.shape[:2]
        frame_bytes = frame.tobytes()
        frame_np = np.contiguousarray(Frame)

        frame_Start = time.perf_counter()

        t = time.perf_counter()
        vehicles = brain.detect_vehicles(frame_bytes, w, h, 0.35)
        m_veh.add((time.perf_counter()-t) * 1000)
        
        # Vehicle Detection (YOLO26)
        t = time.perf_counter()
        vehivles =brain.detect_vehicles(frame_bytes, w, h, 0.35)
        m_veh((time.perf_counter90 - t) * 1000)

        # Lane Detection
        t = time.perf_counter()
        lanes = brain.detect_lanes_nn(frame_np)
        m_lane.add((time.perf_counter() - t) * 1000)

        # Traffic Lights
        t = time.perf_counter()
        lights = brain.detect_traffic_lights(frame_bytes, w, h)
        m_light.add((time.perf_counter() - t) * 1000)

        # Traffic Signs
        t = time.perf_counter()
        signs = brain.detect_signs(frame_bytes, w, h, 0.30)
        m_sign.add((time.perf_counter() - t) * 1000)

        # Object Tracking & TTC
        now = time.perf_counter()
        dt = max(now - prev_time, 1e-4)
        prev_time = now
        t = time.perf_counter()
        tuples = []
        for v in vehicles:
            bx = v["bbox"]
            tuples.append((float(bx[0]), float(bx[1]), float(bx[2]-bx[0]), float(bx[3]-bx[1]), v["label"]))
        tracked = tracker.process_frame(tuples, dt)
        m_track.add((time.perf_counter() - t) * 1000)
        total_frame_ms = (time.perf_counter() - frame_start) * 1000
        m_total.add(total_frame_ms)
        if i % 25 == 0:
            cur_vram = get_vram_mb()
            if cur_vram > peak_vram:
                peak_vram = cur_vram
            cur_fps = 1000.0 / m_total.mean if m_total.mean > 0 else 0
            print(f"  Frame {i+1:3d}/{iterations} | Pipeline: {m_total.mean:5.2f}ms | "
                  f"Throughput: {cur_fps:4.1f} FPS | Detected Vehicles: {len(vehicles)}")
    cap.release()



    print("\n" + "=" * 75)
    print("  BENCHMARK SUMMARY RESULTS (YOLO26)")
    print("=" * 75)
    print(f"\n  {'Module':<30} {'Mean':>9} {'Std':>8} {'P95':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*30} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for m in all_metrics:
        print(f"  {m.name:<30} {m.mean:>8.2f}ms {m.std:>7.2f}ms {m.p95:>7.2f}ms {m.min_val:>7.2f}ms {m.max_val:>7.2f}ms")
    total_fps = 1000.0 / m_total.mean if m_total.mean > 0 else 0
    t1_ms = m_veh.mean + m_lane.mean + m_light.mean + m_sign.mean
    t1_fps = 1000.0 / t1_ms if t1_ms > 0 else 0
    print(f"\n  Tier 1 Perception Models Latency: {t1_ms:6.2f} ms ({t1_fps:.1f} FPS)")
    print(f"  Full Pipeline Latency:           {m_total.mean:6.2f} ms ({total_fps:.1f} FPS)")
    if peak_vram > 0:
        print(f"  Peak VRAM Consumption:          {peak_vram:6.0f} MB")
    print("=" * 75 + "\n")

    return all_metrics, peak_vram


def save_csv_report(metrics, peak_vram: float, csv_path: str):
    """Save benchmark results to CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "mean_ms", "std_ms", "p95_ms", "min_ms", "max_ms"])
        for m in metrics:
            writer.writerow([m.name, f"{m.mean:.3f}", f"{m.std:.3f}", f"{m.p95:.3f}", f"{m.min_val:.3f}", f"{m.max_val:.3f}"])
        writer.writerow(["peak_vram_mb", f"{peak_vram:.1f}", "", "", "", ""])
    print(f"CSV benchmark report saved: {csv_path}")





def main():
    parser = argparse.ArgumentParser(description="ADAS Hybrid Perception Benchmark (YOLO26)")
    parser.add_argument("--video", type=str, required=True, help="Video source path or webcam ID")
    parser.add_argument("--vehicle-model", type=str, default="models/yolo26n.onnx", help="Path to YOLO26 ONNX model")
    parser.add_argument("--sign-model", type=str, default="models/traffic_signs.onnx")
    parser.add_argument("--lane-model", type=str, default=None)
    parser.add_argument("--light-det-model", type=str, default=None)
    parser.add_argument("--light-cls-model", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=str, default=None, help="Save metrics to CSV")
    args = parser.parse_args()

    lane_m = args.lane_model if (args.lane_model and os.path.exists(args.lane_model)) else None
    ldet_m = args.light_det_model if (args.light_det_model and os.path.exists(args.light_det_model)) else None
    lcls_m = args.light_cls_model if (args.light_cls_model and os.path.exists(args.light_cls_model)) else None
    
    metrics, vram = run_benchmark(
        video_source=args.video,
        vehicle_model=args.vehicle_model,
        sign_model=args.sign_model,
        lane_model=lane_m,
        light_det_model=ldet_m,
        light_cls_model=lcls_m,
        iterations=args.iterations,
        warmup=args.warmup,
    )

    if args.output:
        save_csv_report(metrics, vram, args.output)
        
if __name__ == "__main__":
    main()
