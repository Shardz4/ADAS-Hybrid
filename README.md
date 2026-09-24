# ADAS-Hybrid: Edge-Optimized Perception & Intelligence System

A high-performance, hybrid Advanced Driver Assistance System (ADAS) engineered for real-time edge deployment on **NVIDIA Jetson Orin Nano** (8GB) with rapid prototyping and profiling on **RTX 3050 Ti** (4GB).

The architecture separates latency-critical perception (Tier 1 in pure Rust + ONNX Runtime) from high-level reasoning and scene analysis (Tier 2 in Python).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   TIER 1 — Rust ONNX Core (Every Frame)                 │
│                   Target: <10ms on RTX 3050 Ti, <20ms on Orin           │
│                                                                         │
│   Lane Detection    → UFLD-v2 Row-Anchor ONNX + Hough Fallback        │
│   Vehicle Detection → YOLO11n ONNX (PyO3 Zero-Copy Tensor Pipeline)   │
│   Object Tracking   → 2D Kalman Filter + Centroid (Pure Rust CPU)    │
│   Light Detection   → YOLO-Nano Fixture Detector ONNX                 │
│   Light Classify    → EfficientNet-B0 ONNX Crop Classifier            │
│   Light Verify      → 5-Frame Temporal Majority Voter (Zero Cost)     │
│   Sign Recognition  → Custom 4-Class ONNX (Stop, Yield, Speed, etc.)  │
│   Lane Management   → Polyline Smoothing + Ego-Lane Drift / Departure │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ PyO3 Bindings
┌─────────────────────────────────────────────────────────────────────────┐
│                   TIER 2 — Python Intelligence Layer                    │
│                                                                         │
│   Scene Context     → Rule-based heuristics (weather / road density)  │
│   Driving Advisory  → Time-to-Collision (TTC) & Threat Escalation    │
│   [DEV-ONLY] VLM    → SmolVLM-500M (INT4) for A/B quality benchmarking│
│   Sensor Fusion     → Tier 1 conflict resolution & safe overrides     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT & INTERFACES                            │
│                                                                         │
│   Visual HUD        → Real-time OpenCV telemetry overlay             │
│   Audio Alerts      → Non-blocking priority-queued voice advisories   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

##  Hardware Profiles & VRAM Allocation

| Feature | RTX 3050 Ti (Dev & Profiling) | Jetson Orin Nano (Production) |
|---|---|---|
| **Architecture** | Ampere SM 8.6 | Ampere SM 8.7 |
| **VRAM / Memory** | 4GB Dedicated GDDR6 | 8GB Unified LPDDR5 |
| **CUDA Cores** | 2560 | 1024 |
| **DLA Engines** | None (All models on GPU) | 2× Deep Learning Accelerators |
| **Power Target** | ~80W | 7W – 15W |
| **Tier 1 Footprint** | ~1.2GB (FP16) | ~600MB (INT8 TensorRT / DLA) |
| **Optional VLM Slot**| ~2GB (SmolVLM INT4) | Reserved for OS & System Buffers |

---

##  Project Structure

```
ADAS-Hybrid/
├── rust_core/                      # Tier 1: Real-Time Perception Core (Rust)
│   ├── Cargo.toml                  # ort 2.0, pyo3 0.21, ndarray, imageproc, serde
│   └── src/
│       ├── lib.rs                  # PyO3 bindings & AdasBrain multi-model runtime hub
│       ├── kalman.rs               # 2D Constant-Velocity Kalman Filter ([x, y, vx, vy])
│       ├── lane_detect.rs          # UFLD-v2 Row-Anchor decoder + Canny/Hough fallback
│       ├── lane_manager.rs         # Exponential lane smoothing & departure warnings
│       ├── object_proc.rs          # Multi-object Kalman tracker, TTC & distance estimator
│       └── traffic_light.rs        # 2-stage fixture detector, classifier & temporal voter
│
├── app/                            # Tier 2 & Outputs (Python) [In Progress]
│   ├── main.py                     # Main execution pipeline
│   ├── display.py                  # Real-time HUD renderer
│   ├── audio_alert.py              # Priority audio warning engine
│   ├── scene_analyzer.py           # Context & environmental analyzer
│   ├── fusion.py                   # Threat assessment & conflict resolution
│   ├── vlm_engine.py               # Dev-only VLM evaluation engine
│   └── tools/                      # Model export & quantization utilities
│       ├── export_model.py         # ONNX exporter with INT8 quantization
│       ├── export_tensorrt.py      # On-device TensorRT engine builder (with DLA)
│       ├── train_light_detector.py # YOLO-Nano fixture fine-tuning
│       └── train_light_classifier.py# EfficientNet-B0 traffic state classifier
│
├── models/                         # ONNX Model Artifacts [Planned]
│   ├── yolo11n.onnx
│   ├── ufld_culane.onnx
│   ├── light_det.onnx
│   ├── light_cls.onnx
│   └── traffic_signs.onnx
│
├── .gitignore                      # Clean build, model & bytecode ignore rules
└── README.md
```

---

##  Progress & Implementation Status

###  Phase 1: Rust Perception Core (Completed)
- [x] **Kalman Filter Engine (`kalman.rs`)**: 4-state constant velocity motion model, custom matrix algebra (`Mat4`, `Mat2x4`, `Mat4x2`), innovation covariance and gain updates.
- [x] **Lane Perception (`lane_detect.rs`)**: Preprocessing to CHW normalized tensors, UFLD-v2 anchor row classification output decoder, and classic Canny/Hough backup.
- [x] **Lane Stability & Safety (`lane_manager.rs`)**: Alpha-blended polyline temporal smoothing, ego-lane center offset calculation, and drift/departure state machine (`DRIFTING_LEFT`, `DEPARTED_RIGHT`, etc.).
- [x] **Traffic Light Perception (`traffic_light.rs`)**: Two-stage detection architecture (YOLO fixture localization $\rightarrow$ crop extraction $\rightarrow$ softmax state classifier) backed by a 5-frame temporal majority voting filter for VLM-grade reliability at zero runtime cost.
- [x] **Object Processing (`object_proc.rs`)**: Centroid and bounding box association, occlusion ghost-tracking, distance estimation via pinhole camera model, and Time-To-Collision (TTC) calculation.
- [x] **PyO3 Integration Hub (`lib.rs`)**: `AdasBrain`, `Tracker`, and `LaneManager` exported to Python; migrated to `ort 2.0-rc` API with zero-copy array operations.
- [x] **Compilation Verified**: Clean compile on Rust 2021 edition against `ort 2.0.0-rc.13` and `pyo3 0.21.2`.

### Phase 2: Python Orchestration & Intelligence Layer (Next)
- [ ] Pyproject configuration & Maturin build setup
- [ ] `app/main.py` non-blocking main loop
- [ ] `app/display.py` HUD visualization overlay
- [ ] `app/audio_alert.py` priority queue TTS alerts
- [ ] `app/scene_analyzer.py` heuristic context extractor
- [ ] `app/fusion.py` advisory engine & conflict resolution
- [ ] `app/vlm_engine.py` SmolVLM benchmarking hook

###  Phase 3: Export & Training Utilities
- [ ] ONNX export & INT8 quantization scripts
- [ ] LISA / Bosch dataset traffic light fine-tuning scripts
- [ ] Jetson TensorRT + DLA engine compilation scripts

###  Phase 4: Models & Hardware Deployment
- [ ] ONNX weights export & verification
- [ ] Latency benchmarking on RTX 3050 Ti (<10ms target)
- [ ] Jetson Orin Nano cross-compilation & deployment

---

##  Building & Verifying Rust Core

### Prerequisites
- **Rust Toolchain** (1.75+ recommended): `cargo`, `rustc`
- **Python 3.10 - 3.13**
- **CMake** & C++ Build Tools (MSVC on Windows / GCC on Linux)

### Verification
```bash
# Navigate to Rust core
cd rust_core

# Set PyO3 compatibility flag for Python 3.13+ (if applicable)
# Windows PowerShell:
$env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY="1"
# Linux / macOS:
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

# Check compilation
cargo check
```
