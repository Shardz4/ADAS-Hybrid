from dataclasses import dataclass
from typing import List, Dict, Any
from app.scene_analyzer import SceneContext

@dataclass
class Advisory:
    level: str             # "SAFE", "WARNING", "DANGER"
    message: str           # Human-readable HUD banner text
    threats: List[Dict[str, Any]]
    suggested_action: str  # "MAINTAIN_SPEED", "SLOW_DOWN", "BRAKE", "STOP"

class PerceptionFusion:
    """Combines Tier 1 perception outputs and scene context into prioritized advisories."""

    def generate_advisory(self, tier1: dict, context: SceneContext) -> Advisory:
        threats = []
        warning_count = 0
        danger_count = 0

        # Forward collision threats
        for obj in tier1.get("tracked", []):
            # obj tuple: (id, x, y, w, h, distance, speed, ttc, vx, vy, label)
            obj_id, _, _, _, _, dist, speed, ttc, _, _, label = obj
            
            if ttc < 2.0 and speed > 0.5:
                danger_count += 1
                threats.append({
                    "type": "COLLISION_IMMINENT",
                    "id": obj_id,
                    "label": label,
                    "ttc": ttc,
                    "dist": dist,
                    "severity": "DANGER",
                })
            elif ttc < 3.8 and speed > 0.2:
                warning_count += 1
                threats.append({
                    "type": "COLLISION_WARNING",
                    "id": obj_id,
                    "label": label,
                    "ttc": ttc,
                    "dist": dist,
                    "severity": "WARNING",
                })

        # Lane departure status
        departure = tier1.get("departure", "NONE")
        if departure in ("DEPARTED_LEFT", "DEPARTED_RIGHT"):
            danger_count += 1
            threats.append({"type": "LANE_DEPARTURE", "status": departure, "severity": "DANGER"})
        elif departure in ("DRIFTING_LEFT", "DRIFTING_RIGHT"):
            warning_count += 1
            threats.append({"type": "LANE_DRIFT", "status": departure, "severity": "WARNING"})

        # Traffic light
        for light in tier1.get("lights", []):
            voted = light.get("voted_status", "None")
            if voted == "Red":
                warning_count += 1
                threats.append({"type": "RED_LIGHT", "severity": "WARNING"})
            elif voted == "Yellow":
                threats.append({"type": "YELLOW_LIGHT", "severity": "INFO"})

        # Traffic signs
        for sign in tier1.get("signs", []):
            cid = sign.get("class_id", -1)
            if cid == 0:  # Stop sign
                warning_count += 1
                threats.append({"type": "STOP_SIGN", "severity": "WARNING"})

        # advisory
        if danger_count > 0 or warning_count >= 2:
            level = "DANGER"
            suggested_action = "BRAKE"
            if threats and threats[0]["type"] == "COLLISION_IMMINENT":
                message = f"EMERGENCY BRAKE: {threats[0]['label'].upper()} IN PATH ({threats[0]['ttc']:.1f}s)"
            elif departure.startswith("DEPARTED"):
                message = "CRITICAL: LANE DEPARTURE DETECTED"
            else:
                message = "DANGER: MULTIPLE HAZARDS DETECTED"
        elif warning_count == 1:
            level = "WARNING"
            suggested_action = "SLOW_DOWN"
            top_threat = threats[0]
            if top_threat["type"] == "COLLISION_WARNING":
                message = f"CAUTION: Closing on {top_threat['label']} ({top_threat['ttc']:.1f}s TTC)"
            elif top_threat["type"] == "LANE_DRIFT":
                message = f"WARNING: {top_threat['status'].replace('_', ' ')}"
            elif top_threat["type"] == "RED_LIGHT":
                message = "CAUTION: Red Traffic Light Ahead"
            elif top_threat["type"] == "STOP_SIGN":
                message = "CAUTION: Stop Sign Ahead"
            else:
                message = "WARNING: Hazard Ahead"
        else:
            level = "SAFE"
            suggested_action = "MAINTAIN_SPEED"
            message = "PATH CLEAR"

        return Advisory(
            level=level,
            message=message,
            threats=threats,
            suggested_action=suggested_action,
        )