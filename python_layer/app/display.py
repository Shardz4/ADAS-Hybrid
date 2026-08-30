import cv2
import numpy as np

class HudRenderer:
    def render(
        self,
        frame: np.ndarray,
        tier1: dict,
        context,
        advisory,
        fps: float,
        vlm_text: str = None,
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        departure = tier1.get("departure", "NONE")
        if departure.startswith("DEPARTED"):
            lane_color = (0, 0, 255)    
        elif departure.startswith("DRIFTING"):
            lane_color = (0, 215, 255)    
        else:
            lane_color = (0, 255, 0)    

        lanes = tier1.get("lanes", [])
        for lane in lanes:
            if isinstance(lane, list) and len(lane) >= 2:
                # Polylines from UFLD
                pts = np.array(lane, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=lane_color, thickness=3)
            elif isinstance(lane, (list, tuple)) and len(lane) == 2:
                # Hough line endpoint pair ((x1, y1), (x2, y2))
                pt1 = (int(lane[0][0]), int(lane[0][1]))
                pt2 = (int(lane[1][0]), int(lane[1][1]))
                cv2.line(frame, pt1, pt2, lane_color, 3)

       
        for obj in tier1.get("tracked", []):
            # obj: (id, x, y, w, h, distance, speed, ttc, vx, vy, label)
            obj_id, ox, oy, ow, oh, dist, _, ttc, _, _, label = obj
            x1, y1 = int(ox), int(oy)
            x2, y2 = int(ox + ow), int(oy + oh)

            if ttc < 2.0:
                box_color = (0, 0, 255)
            elif ttc < 4.0:
                box_color = (0, 215, 255)
            else:
                box_color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            tag = f"ID:{obj_id} {label} {dist:.1f}m"
            if ttc < 90.0:
                tag += f" | {ttc:.1f}s"

            cv2.putText(
                frame,
                tag,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2,
                cv2.LINE_AA,
            )

    
        lights = tier1.get("lights", [])
        if lights:
            voted_status = lights[0].get("voted_status", "None")
            badge_color = (70, 70, 70)
            if voted_status == "Red":
                badge_color = (0, 0, 255)
            elif voted_status == "Yellow":
                badge_color = (0, 215, 255)
            elif voted_status == "Green":
                badge_color = (0, 255, 0)

            cv2.circle(frame, (45, 45), 20, badge_color, -1)
            cv2.circle(frame, (45, 45), 20, (255, 255, 255), 2)
            cv2.putText(
                frame,
                voted_status.upper(),
                (75, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        for sign in tier1.get("signs", []):
            bx = sign.get("bbox", [0, 0, 0, 0])
            sx1, sy1, sx2, sy2 = map(int, bx)
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (255, 128, 0), 2)
            cv2.putText(
                frame,
                f"SIGN {sign.get('class_id', '')}",
                (sx1, max(15, sy1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 128, 0),
                1,
                cv2.LINE_AA,
            )

        ctx_str = f"FPS: {fps:.1f} | {context.time_of_day.upper()} | {context.weather.upper()} | {context.road_type.upper()}"
        cv2.putText(
            frame,
            ctx_str,
            (w - 480, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        banner_h = 44
        overlay = frame.copy()
        if advisory.level == "DANGER":
            bg_color = (0, 0, 180)
        elif advisory.level == "WARNING":
            bg_color = (0, 140, 220)
        else:
            bg_color = (30, 30, 30)

        cv2.rectangle(overlay, (0, h - banner_h), (w, h), bg_color, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        adv_text = f"[{advisory.level}] {advisory.message} | ACTION: {advisory.suggested_action}"
        cv2.putText(
            frame,
            adv_text,
            (20, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

      
        if vlm_text:
            cv2.putText(
                frame,
                vlm_text,
                (20, h - banner_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 255, 200),
                1,
                cv2.LINE_AA,
            )

        return frame