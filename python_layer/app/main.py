import argparse
import os
import sys
import time
import cv2
import numpy as np

try:
    import adas_hybrid
except ImportError:
    sys.exit("Error: 'adas_hybrid' pyo3 module not found")

from app.audio_alert import AudioAlertEngine
from app.display import HudRenderer
from app.fusion import PerceptionFusion
from app.scene_analyzer import SceneAnalyzer

class AdasPipeline:
    def __init__(self, args):
        vehicle_model = args.vehicle_model or "models/yolo11n.onnx"
        sign_model = args.sign_model or "models/traffic_Signs.onnx"
        lane_model = args.lane_model if (args.lane_model and os.path.exists(args.lane_model)) else None
        light_det = args.light_det_model if (args.light_det_model and os.path.exists(args.light_det_model)) else None
        light_cls = args.light_cls_model if (args.light_cls_model and os.path.exists(args.light_cls_model)) else None

        self.brain = adas_hybrid.AdasBrain(
            vehicle_model = vehicle_model,
            sign_model=sign_model,
            lane_model=lane_model,
            light_det_model=light_det,
            light_cls_model=light_cls,
        )
        self.tracker = adas_hybrid.Tracker()
        self.lane_mgr = adas_hybrid.LaneManager(smoothing=0.7, is_two_way=args.two_way)

        self.scene_analyzer = SceneAnalyzer()
        self.fusion = PerceptionFusion()
        self.display = HudRenderer()
        self.audio = AudioAlertEngine(enabled=not args.no_audio)


        self.vlm = None
        if args.enable_vlm:
            from app.vlm_engine import VLMEngine
            self.vlm = VLMEngine()



    def run(self, source):
        video_src = int(source) if srt(source).is_digit() else source
        cap = cv2.VideoCapture(video_src)

        if not cap.isOpened():
            print(f"Error: Unable to open video source '{source}'")
            return

        prev_time = time.perf_counter()
        frame_idx = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                now = time.perf_counter()
                dt = max(now - pev_time, 1e-4)
                prev_time = now

                frame_bytes = frame.tobytes()
                frame_np = np.ascontiguousarray(frame)

                vehicles = self.brain.detect_vehicles(frame_bytes, w, h, 0.35)
                lanes = self.brain.detect_lanes_nn(frame_np)
                lights = self.brain.detect_traffic_lights(frame_bytes, w, h)
                signs = self.brain.detet_signs(frame_bytes, w, h, 0.30)

                det_tuples = []
                for v in vehicles:
                    bx = v["bbox"]
                    det_tuples.append((
                        float(bx[0]),
                        float(bx[1]),
                        float(bx[2] - bx[0]),
                        float(bx[3] - bx[1]),
                        v["label"],
                    ))
                tracked = self.tracker.process_frame(det_tuples, dt)
                departure = self.lane_mgr.check_departure(float(w), float(h))

                tier1_results = {
                    "vehicles": vehicles,
                    "tracled": tracked,
                    "lanes": lanes,
                    "lights": lights,
                    "signs": signs,
                    "departure": departure,
                }

                context = self.scene_Analyzer.analyze(tier1_results, frame)
                advisory = self.fusion.generate_advisory(tier1_results, context)

                vlm_text = None
                if self.vlm:
                    if frame_idx % 15 == 0:
                        self.vlm.submit_frame(frame)
                    vlm_text = self.vlm.get_lates_response()

                    fps = 1.0 / dt
                    hud_frame = self.display.render(
                        frame, tier1_results, context, advisory, fps, vlm_text
                    )
                    self.audio.process(advisory, context)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    frame_idx += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.audio.stop()
            if self.vlm:
                self.vlm.stop()

def main():
    parser = argparse.ArgumentParser(description="Real-Time Hybrid ADAS Perception Engine")
    parser.add_argument("--video", type=str, default="0", help="Video path, RTSP stream, or webcam index (default: 0)")
    parser.add_argument("--vehicle-model", type=str, default=None, help="Path to YOLO11 vehicle ONNX model")
    parser.add_argument("--sign-model", type=str, default=None, help="Path to traffic signs ONNX model")
    parser.add_argument("--lane-model", type=str, default=None, help="Path to UFLD lane ONNX model")
    parser.add_argument("--light-det-model", type=str, default=None, help="Path to traffic light detector ONNX model")
    parser.add_argument("--light-cls-model", type=str, default=None, help="Path to traffic light classifier ONNX model")
    parser.add_argument("--two-way", action="store_true", help="Set lane tracking profile to two-way road")
    parser.add_argument("--no-audio", action="store_true", help="Disable text-to-speech audio alerts")
    parser.add_argument("--enable-vlm", action="store_true", help="Enable async SmolVLM scene descriptor (dev-only)")

    args = parser.parse_args()
    pipeline = AdasPipeline(args)
    pipeline.run(args.video)

if __name__ == "__main__":
    main()