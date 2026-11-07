import math
import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from config import (
    MP_DETECTION_CONFIDENCE,
    MP_PRESENCE_CONFIDENCE,
    MP_TRACKING_CONFIDENCE,
    ROI_PADDING,
)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


class GestureController:
    def __init__(self):
        self.use_tasks_api = not hasattr(mp, "solutions")

        if not self.use_tasks_api:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=MP_DETECTION_CONFIDENCE,
                min_tracking_confidence=MP_TRACKING_CONFIDENCE,
            )
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            if not os.path.exists(MODEL_PATH):
                print("[GestureController] Downloading hand_landmarker.task...")
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

            base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=MP_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=MP_PRESENCE_CONFIDENCE,
                min_tracking_confidence=MP_TRACKING_CONFIDENCE,
            )
            self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

        self.palm_hold_frames = 0
        self.pinch_hold_frames = 0

    def _detect_hands(self, rgb_frame):
        if not self.use_tasks_api:
            results = self.hands.process(rgb_frame)
            if not results.multi_hand_landmarks:
                return [], []
            lms_list = [hl.landmark for hl in results.multi_hand_landmarks]
            labels = []
            for idx in range(len(lms_list)):
                if results.multi_handedness and idx < len(results.multi_handedness):
                    labels.append(results.multi_handedness[idx].classification[0].label)
                else:
                    labels.append("Hand")
            return lms_list, labels
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.landmarker.detect(mp_image)
            if not result.hand_landmarks:
                return [], []
            lms_list = result.hand_landmarks
            labels = []
            for idx in range(len(lms_list)):
                if result.handedness and idx < len(result.handedness):
                    labels.append(result.handedness[idx][0].category_name or "Hand")
                else:
                    labels.append("Hand")
            return lms_list, labels

    def _compute_palm_normal(self, lms) -> np.ndarray:
        """Compute 3D unit normal vector of the palm using Landmarks 0, 5, and 17."""
        p0 = np.array([lms[0].x, lms[0].y, lms[0].z], dtype=np.float32)
        p5 = np.array([lms[5].x, lms[5].y, lms[5].z], dtype=np.float32)
        p17 = np.array([lms[17].x, lms[17].y, lms[17].z], dtype=np.float32)
        v1 = p17 - p0
        v2 = p5 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def _detect_fruit_in_scan_zone(self, raw_frame) -> tuple:
        """
        Fallback detector: checks if a colorful fruit (yellow, red, green, orange)
        is held inside the center scan zone even when fingers are occluded.
        """
        h, w, _ = raw_frame.shape
        x1, y1 = int(w * 0.28), int(h * 0.20)
        x2, y2 = int(w * 0.72), int(h * 0.80)
        zone = raw_frame[y1:y2, x1:x2]
        if zone.size == 0:
            return False, None, None

        hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
        # Saturation > 65 and Value > 55 captures vibrant fruits (banana, apple, orange, mango, tomato)
        fruit_mask = cv2.inRange(hsv, (0, 68, 55), (180, 255, 255))
        kernel = np.ones((5, 5), np.uint8)
        fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, None, None

        largest = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(largest) / float(zone.shape[0] * zone.shape[1])
        if area_ratio < 0.08:
            return False, None, None

        fx, fy, fw, fh = cv2.boundingRect(largest)
        pad = 25
        bx1 = max(0, x1 + fx - pad)
        by1 = max(0, y1 + fy - pad)
        bx2 = min(w, x1 + fx + fw + pad)
        by2 = min(h, y1 + fy + fh + pad)
        crop = raw_frame[by1:by2, bx1:bx2].copy()
        return True, crop, (bx1, by1, bx2, by2)

    def process_frame(self, frame):
        """
        Returns dict with:
        - annotated_frame
        - target_present: bool (True if hand or fruit is inside scan zone)
        - centered: bool
        - centering_hint: str or None
        - primary_crop: np.ndarray or None
        - palm_normal: np.ndarray or None
        - instant_gesture: None | 'PINCH_SCAN' | 'PALM_RECAP'
        - hand_boxes: list of (x1, y1, x2, y2, label)
        """
        raw_frame = frame.copy()
        annotated = frame.copy()
        h, w, _ = annotated.shape

        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        lms_list, labels = self._detect_hands(rgb)

        target_present = False
        centered = False
        centering_hint = None
        primary_crop = None
        palm_normal = None
        instant_gesture = None
        hand_boxes = []

        if lms_list:
            for idx, lms in enumerate(lms_list):
                label = labels[idx]
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
                for p1, p2 in HAND_CONNECTIONS:
                    cv2.line(annotated, pts[p1], pts[p2], (0, 230, 140), 2)
                for px, py in pts:
                    cv2.circle(annotated, (px, py), 4, (255, 255, 255), -1)

                xs = [lm.x for lm in lms]
                ys = [lm.y for lm in lms]
                x1 = max(0, int((min(xs) - ROI_PADDING) * w))
                y1 = max(0, int((min(ys) - ROI_PADDING) * h))
                x2 = min(w, int((max(xs) + ROI_PADDING) * w))
                y2 = min(h, int((max(ys) + ROI_PADDING) * h))
                crop = raw_frame[y1:y2, x1:x2].copy()
                hand_boxes.append((x1, y1, x2, y2, f"{label} + Fruit ROI"))

                if idx == 0:
                    target_present = True
                    primary_crop = crop
                    palm_normal = self._compute_palm_normal(lms)
                    cx, cy = lms[9].x, lms[9].y

                    if cx < 0.22:
                        centering_hint = "Move hand toward the center."
                    elif cx > 0.78:
                        centering_hint = "Move hand toward the center."
                    elif cy < 0.18:
                        centering_hint = "Lower your hand slightly."
                    elif cy > 0.82:
                        centering_hint = "Raise your hand slightly."
                    else:
                        centered = True

                    # Check Pinch (Thumb #4 and Index #8)
                    pinch_dist = math.hypot(lms[4].x - lms[8].x, lms[4].y - lms[8].y)
                    tx, ty = int(lms[4].x * w), int(lms[4].y * h)
                    ix, iy = int(lms[8].x * w), int(lms[8].y * h)
                    cv2.line(annotated, (tx, ty), (ix, iy), (0, 255, 100) if pinch_dist < 0.055 else (0, 200, 255), 3)

                    if pinch_dist < 0.055:
                        self.pinch_hold_frames += 1
                        if self.pinch_hold_frames == 8:
                            instant_gesture = "PINCH_SCAN"
                    else:
                        self.pinch_hold_frames = 0

                    # Check Open Palm (4 fingers upright & spread for Pantry Recap)
                    tips = [8, 12, 16, 20]
                    pips = [6, 10, 14, 18]
                    ext = sum(1 for t, p in zip(tips, pips) if lms[t].y < lms[p].y)
                    if ext == 4 and pinch_dist > 0.12:
                        self.palm_hold_frames += 1
                        if self.palm_hold_frames == 16:
                            instant_gesture = "PALM_RECAP"
                    else:
                        self.palm_hold_frames = 0
        else:
            self.pinch_hold_frames = 0
            self.palm_hold_frames = 0

            # Fallback: Check if a fruit is held directly inside the center scan zone
            fruit_found, fruit_crop, fruit_bbox = self._detect_fruit_in_scan_zone(raw_frame)
            if fruit_found:
                target_present = True
                centered = True
                primary_crop = fruit_crop
                bx1, by1, bx2, by2 = fruit_bbox
                hand_boxes.append((bx1, by1, bx2, by2, "Detected Fruit ROI"))

        return {
            "annotated_frame": annotated,
            "target_present": target_present,
            "centered": centered,
            "centering_hint": centering_hint,
            "primary_crop": primary_crop,
            "palm_normal": palm_normal,
            "instant_gesture": instant_gesture,
            "hand_boxes": hand_boxes,
        }

    def close(self):
        if not self.use_tasks_api:
            self.hands.close()
        else:
            self.landmarker.close()