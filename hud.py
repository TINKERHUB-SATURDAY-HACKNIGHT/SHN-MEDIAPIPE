import cv2
import numpy as np


class HUD:
    URGENCY_COLORS = {
        "EAT_TODAY": (40, 60, 255),     # Bright Red-Orange (Use First)
        "USE_SOON": (0, 215, 255),      # Gold-Yellow (Use Second)
        "STORE_LONGER": (80, 235, 100), # Bright Green (Use Last)
    }

    URGENCY_TAGS = {
        "EAT_TODAY": "USE FIRST (Today)",
        "USE_SOON": "USE SOON (2-3d)",
        "STORE_LONGER": "USE LAST (Keep)",
    }

    def render(
        self,
        frame: np.ndarray,
        detection: dict,
        state_name: str,
        elapsed_10s: float,
        rotation_deg: float,
        caption: str,
        pantry_items: list,
        is_flipped: bool,
    ) -> np.ndarray:
        h, w, _ = frame.shape
        canvas = frame.copy()

        # 1. Center Scan Zone Box
        zx1, zy1 = int(w * 0.28), int(h * 0.20)
        zx2, zy2 = int(w * 0.72), int(h * 0.80)
        zone_color = (0, 255, 140) if detection["centered"] else (180, 180, 180)
        cv2.rectangle(canvas, (zx1, zy1), (zx2, zy2), zone_color, 2)
        cv2.putText(
            canvas,
            "360 FRUIT SCAN ZONE (Hold Fruit Here)",
            (zx1 + 12, zy1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            zone_color,
            2,
        )

        # 2. Draw Hand / Fruit Bounding Boxes
        for x1, y1, x2, y2, label in detection["hand_boxes"]:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 200), 2)
            cv2.putText(
                canvas,
                label,
                (x1, max(85, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 200),
                2,
            )

        # 3. Top Header Bar
        cv2.rectangle(canvas, (0, 0), (w, 72), (18, 18, 18), -1)
        cv2.putText(
            canvas,
            f"FreshPick 360 | State: {state_name}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (0, 255, 180),
            2,
        )
        cam_mode = "Mirror" if is_flipped else "Normal Selfie"
        controls = f"SPACE/Pinch: Instant Scan | R/Open Palm: Priority Recap | F: Camera ({cam_mode}) | F11: Fullscreen | Q: Quit"
        cv2.putText(
            canvas,
            controls,
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 220, 220),
            1,
        )

        # 4. 10-Second Guided Cycle Progress Bar
        if elapsed_10s > 0.05:
            bar_x1, bar_y1, bar_x2, bar_y2 = 20, 78, w - 20, 102
            cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (45, 45, 45), -1)
            fill_w = int((bar_x2 - bar_x1) * min(1.0, elapsed_10s / 10.0))
            cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), (0, 230, 120), -1)
            timer_label = f"10s Guided Cycle: {elapsed_10s:.1f}s / 10.0s"
            if rotation_deg > 1.0:
                timer_label += f" | Wrist Rotation: {int(rotation_deg)} deg"
            cv2.putText(
                canvas,
                timer_label,
                (bar_x1 + 12, bar_y2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        # 5. Right Side Panel: USE FIRST -> USE LAST Queue
        if pantry_items:
            pw = 360
            px1 = max(10, w - pw - 16)
            py1 = 114
            py2 = py1 + 42 + min(6, len(pantry_items)) * 30
            cv2.rectangle(canvas, (px1, py1), (w - 16, py2), (22, 22, 22), -1)
            cv2.rectangle(canvas, (px1, py1), (w - 16, py2), (100, 100, 100), 1)
            cv2.putText(
                canvas,
                "PRIORITY ORDER (USE FIRST -> LAST):",
                (px1 + 12, py1 + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (255, 255, 255),
                2,
            )
            for idx, item in enumerate(pantry_items[:6]):
                urg = item["urgency"]
                color = self.URGENCY_COLORS.get(urg, (200, 200, 200))
                tag = self.URGENCY_TAGS.get(urg, urg)
                line = f"{idx + 1}. {item['fruit_name']} -> {tag}"
                cv2.putText(
                    canvas,
                    line[:40],
                    (px1 + 12, py1 + 54 + idx * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    color,
                    2,
                )

        # 6. Bottom High-Contrast Voice Caption Banner
        cv2.rectangle(canvas, (0, h - 92), (w, h), (15, 15, 15), -1)
        lines = self._wrap_text(caption, max_chars=95)
        for i, line in enumerate(lines[:2]):
            cv2.putText(
                canvas,
                line,
                (20, h - 54 + i * 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 245, 255),
                2,
            )

        return canvas

    @staticmethod
    def _wrap_text(text: str, max_chars: int = 95) -> list:
        words = str(text).split()
        lines, current = [], ""
        for word in words:
            if len(current) + len(word) + 1 <= max_chars:
                current = f"{current} {word}".strip()
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines if lines else ["Ready."]