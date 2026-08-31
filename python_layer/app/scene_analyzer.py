from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SceneContext:
    time_of_day: str  #"day, "night", "dawn/dusk"
    weather: str   # "clear"
    road_type: str 
    congestion: str
    hazard_level: str
    vehicle_count: int
    avg_ttc: float  #ttc - time to collision
    min_ttc: float
    lane_confidence: str

class SceneAnalyzer:
    """scene context analyzer"""
    def analyze(self, tier1: dict, frame: np.ndarray) -> SceneContext:
        h, w = frame.shape[:2]

        sample = frame[::4, ::4]
        gray_sample = np.mean(sample, axis = 2) if sample.ndim == 3 else sample

        upper_third = gray_sample[: sample.shape[0] // 3, :]
        mean_brightness = float(np.mean(upper_third))

        if mean_brightness < 50.0:
            time_of_day = "night"
        elif mean_brightness < 115.0:
            time_of_day = "dawn/dusk"
        else:
            time_of_day = "day"

        #weather / visibiliy
        std_dev = float(np.std(gray_sample))
        if std_dev < 28.0 and mean_brightness < 120.0:
            weather = "fog"
        else:
            weather = "clear"

        tracked = tier1.get("tracked", [])
        vehicle_count = len(tracked)

        ttc_list = [t[7] for t in tracked if t[7] < 90.0]
        min_ttc = min(ttc_list) if ttc_list else 99.0
        avg_ttc = float(np.mean(ttc_list)) if ttc_list else 99.0

        # Road type

        lanes = tier1.get("lanes", [])
        lane_count = len(lanes)

        if lane_count >= 3:
            road_type = "highway"
        elif vehicle_count >= 4 or (tier1.get("lights") and len(tier1.get("lights")) > 0):
            road_type = "urban"
        else:
            road_type = "rural"

        if vehicle_count > 6:
            congestion = "heavy"
        elif vehicle_count >=3:
            congestion = "moderate"
        else:
            congestion = "free_flow"

        #lane_conf

        if lane_count >= 2:
            lane_confidence = "strong"
        elif lane_count == 1:
            lane_confidence = "weak"
        else:
            lane_confidence = "none"

        #hazard level

        if min_ttc < 1.8 or tier1.get("departure") in ("DEPARTED_LEFT", "DEPARTED_RIGHT"):
            hazard_level = "critical"
        elif min_ttc< 3.2 or tier1.get("departure") in ("DRIFTING_LEFT", "DRIFTING_RIGHT"):
            hazard_level = "high"
        elif min_ttc < 5.0:
            hazard_level = "medium"
        else:
            hazard_level = "low"

        return SceneContext(
            time_of_day=time_of_day,
            weather=weather,
            road_type=road_type,
            congestion=congestion,
            hazard_level=hazard_level,
            vehicle_count=vehicle_count,
            avg_ttc=avg_ttc,
            min_ttc=min_ttc,
            lane_confidence=lane_confidence,
        )