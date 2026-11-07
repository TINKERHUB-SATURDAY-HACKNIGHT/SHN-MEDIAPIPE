import cv2
import numpy as np


class HUD:
    URGENCY_COLORS = {
        "EAT_TODAY": (0, 80, 255),      # Red-Orange (BGR)
        "USE_SOON": (0, 215, 255),      # Gold-Yellow (BGR)
        "STORE_LONGER": (80, 220, 80),  # Green (BGR)
    }

    def render(
        self,
        frame: np.ndarray,
        hand_data: list,
        gesture_progress: float,
        last_caption: str,
        pantry_items: list,
        status_text: str = "Pinch: Scan | 2 Hands: Compare | Open Palm: Pantry List",
    ) -> np.ndarray:
        h, w, _ = frame.shape
        canvas = frame.copy()

        # 1. Center Sweet-Spot Guide Box
        cx1, cy1 = int(w * 0.25), int(h * 0.20)
        cx2, cy2 = int(w * 0.75), int(h * 0.80)
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (180, 180, 180), 1)
        cv2.putText(
            canvas,
            "SCAN ZONE",
            (cx1 + 8, cy1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # 2. Hand / Fruit ROI Bounding Boxes
        for hand in hand_data:
            x1, y1, x2, y2 = hand["bbox"]
            label = hand["label"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 180), 2)
            cv2.putText(
                canvas,
                f"{label} Fruit ROI",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 180),
                2,
            )

        # 3. Top Header Banner
        cv2.rectangle(canvas, (0, 0), (w, 62), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            "FreshPick | Assistive Gesture Freshness & Image-to-Text",
            (14, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 255, 180),
            2,
        )
        cv2.putText(
            canvas,
            status_text,
            (14, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
        )

        # 4. Gesture Hold Progress Bar
        if gesture_progress > 0.05:
            bar_w = int((w - 28) * min(1.0, gesture_progress))
            cv2.rectangle(canvas, (14, 68), (w - 14, 82), (50, 50, 50), -1)
            cv2.rectangle(canvas, (14, 68), (14 + bar_w, 82), (0, 255, 120), -1)

        # 5. Pantry Priority Mini-Panel (Top Right)
        if pantry_items:
            panel_x = max(10, w - 270)
            panel_h = 34 + min(4, len(pantry_items)) * 24
            cv2.rectangle(canvas, (panel_x, 90), (w - 10, 90 + panel_h), (25, 25, 25), -1)
            cv2.putText(
                canvas,
                "USE FIRST QUEUE:",
                (panel_x + 10, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            for idx, item in enumerate(pantry_items[:4]):
                color = self.URGENCY_COLORS.get(item["urgency"], (200, 200, 200))
                text = f"{idx + 1}. {item['fruit_name']} [{item['urgency']}]"
                cv2.putText(
                    canvas,
                    text[:32],
                    (panel_x + 10, 136 + idx * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    color,
                    1,
                )

        # 6. High-Contrast Bottom Caption Bar (Image-to-Text Output)
        cv2.rectangle(canvas, (0, h - 78), (w, h), (15, 15, 15), -1)
        wrapped_lines = self._wrap_text(last_caption, max_chars=78)
        for i, line in enumerate(wrapped_lines[:2]):
            cv2.putText(
                canvas,
                line,
                (14, h - 46 + (i * 26)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (0, 245, 255),
                2,
            )

        return canvas

    @staticmethod
    def _wrap_text(text: str, max_chars: int = 78) -> list:
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