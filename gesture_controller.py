import math
import os
import time
import urllib.request
import cv2
import mediapipe as mp

# Official MediaPipe HandLandmarker model URL (auto-downloaded once if using Tasks API)
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
            # Legacy MediaPipe (<= 0.10.21)
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            )
        else:
            # Modern MediaPipe Tasks API (0.10.31+)
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            if not os.path.exists(MODEL_PATH):
                print("[GestureController] Downloading hand_landmarker.task model (~7.5 MB)...")
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                print("[GestureController] Model downloaded successfully.")

            base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.65,
                min_hand_presence_confidence=0.65,
                min_tracking_confidence=0.65,
            )
            self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

        # Hold counters to avoid accidental triggers
        self.pinch_frames = 0
        self.PINCH_REQUIRED = 12       # ~0.4s hold

        self.palm_frames = 0
        self.PALM_REQUIRED = 18        # ~0.6s hold

        self.dual_frames = 0
        self.DUAL_REQUIRED = 25        # ~0.85s hold with 2 hands

        self.thumb_frames = 0
        self.THUMB_REQUIRED = 16

        self.last_trigger_time = 0.0
        self.cooldown_sec = 3.5        # Cooldown between scans

    def _draw_hand_skeleton(self, frame, lms):
        """Draw 21 hand landmarks and connections using pure OpenCV."""
        h, w, _ = frame.shape
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(frame, pts[p1], pts[p2], (0, 220, 150), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)

    def _count_extended_fingers(self, lms) -> int:
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        extended = 0
        for tip, pip in zip(tips, pips):
            if lms[tip].y < lms[pip].y:
                extended += 1
        return extended

    def _is_thumbs_up(self, lms) -> bool:
        thumb_tip = lms[4]
        thumb_ip = lms[3]
        index_mcp = lms[5]
        other_fingers = self._count_extended_fingers(lms)
        return (thumb_tip.y < thumb_ip.y < index_mcp.y) and (other_fingers == 0)

    def _extract_hand_crop(self, raw_frame, lms, pad: float = 0.14):
        h, w, _ = raw_frame.shape
        xs = [lm.x for lm in lms]
        ys = [lm.y for lm in lms]

        x1 = max(0, int((min(xs) - pad) * w))
        y1 = max(0, int((min(ys) - pad) * h))
        x2 = min(w, int((max(xs) + pad) * w))
        y2 = min(h, int((max(ys) + pad) * h))

        crop = raw_frame[y1:y2, x1:x2].copy()
        return crop, (x1, y1, x2, y2)

    def _detect_hands(self, rgb_frame):
        """Unified detector returning (list_of_21_landmarks, list_of_hand_labels)."""
        if not self.use_tasks_api:
            results = self.hands.process(rgb_frame)
            if not results.multi_hand_landmarks:
                return [], []
            landmarks_list = [hl.landmark for hl in results.multi_hand_landmarks]
            labels = []
            for idx in range(len(landmarks_list)):
                if results.multi_handedness and idx < len(results.multi_handedness):
                    labels.append(results.multi_handedness[idx].classification[0].label)
                else:
                    labels.append("Hand")
            return landmarks_list, labels
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.landmarker.detect(mp_image)
            if not result.hand_landmarks:
                return [], []
            landmarks_list = result.hand_landmarks
            labels = []
            for idx in range(len(landmarks_list)):
                if result.handedness and idx < len(result.handedness):
                    labels.append(result.handedness[idx][0].category_name or "Hand")
                else:
                    labels.append("Hand")
            return landmarks_list, labels

    def process_frame(self, frame):
        raw_frame = frame.copy()
        annotated = frame.copy()
        h, w, _ = annotated.shape

        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        landmarks_list, labels = self._detect_hands(rgb)

        gesture_event = None
        gesture_progress = 0.0
        hand_data = []
        centering_hint = None

        now = time.time()
        in_cooldown = (now - self.last_trigger_time) < self.cooldown_sec

        if landmarks_list:
            num_hands = len(landmarks_list)

            for idx, lms in enumerate(landmarks_list):
                label = labels[idx]
                self._draw_hand_skeleton(annotated, lms)

                crop, bbox = self._extract_hand_crop(raw_frame, lms)
                middle_mcp = lms[9]
                wrist = lms[0]

                hand_data.append({
                    "label": label,
                    "crop": crop,
                    "bbox": bbox,
                    "center": (middle_mcp.x, middle_mcp.y),
                })

                # Spatial audio guidance for visually impaired framing (primary hand)
                if idx == 0:
                    hand_size = math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y)
                    if hand_size < 0.09:
                        centering_hint = "Bring the fruit slightly closer to the camera."
                    elif middle_mcp.x < 0.22 or middle_mcp.x > 0.78:
                        centering_hint = "Move your hand slightly toward the center."
                    elif middle_mcp.y < 0.20:
                        centering_hint = "Lower your hand slightly."
                    elif middle_mcp.y > 0.82:
                        centering_hint = "Raise your hand slightly."

                # 1. Measure Pinch (Thumb tip #4 to Index tip #8)
                thumb = lms[4]
                index = lms[8]
                pinch_dist = math.hypot(thumb.x - index.x, thumb.y - index.y)

                tx, ty = int(thumb.x * w), int(thumb.y * h)
                ix, iy = int(index.x * w), int(index.y * h)
                is_pinching = pinch_dist < 0.048
                cv2.line(
                    annotated,
                    (tx, ty),
                    (ix, iy),
                    (0, 255, 120) if is_pinching else (0, 200, 255),
                    3,
                )

                if not in_cooldown:
                    if is_pinching:
                        self.pinch_frames += 1
                        gesture_progress = max(
                            gesture_progress,
                            min(1.0, self.pinch_frames / self.PINCH_REQUIRED),
                        )
                        if self.pinch_frames >= self.PINCH_REQUIRED:
                            gesture_event = "PINCH_SCAN"
                            self._reset_counters()
                            self.last_trigger_time = now
                            break
                    elif idx == 0:
                        self.pinch_frames = max(0, self.pinch_frames - 1)

                    # 2. Check Open Palm (4 fingers extended & not pinching)
                    extended_count = self._count_extended_fingers(lms)
                    if extended_count == 4 and pinch_dist > 0.11 and num_hands == 1:
                        self.palm_frames += 1
                        gesture_progress = max(
                            gesture_progress,
                            min(1.0, self.palm_frames / self.PALM_REQUIRED),
                        )
                        if self.palm_frames >= self.PALM_REQUIRED:
                            gesture_event = "PALM_RECAP"
                            self._reset_counters()
                            self.last_trigger_time = now
                            break
                    elif idx == 0:
                        self.palm_frames = max(0, self.palm_frames - 1)

                    # 3. Check Thumbs Up (Tactile & Olfactory safety tip)
                    if self._is_thumbs_up(lms) and num_hands == 1:
                        self.thumb_frames += 1
                        gesture_progress = max(
                            gesture_progress,
                            min(1.0, self.thumb_frames / self.THUMB_REQUIRED),
                        )
                        if self.thumb_frames >= self.THUMB_REQUIRED:
                            gesture_event = "THUMB_SAFETY"
                            self._reset_counters()
                            self.last_trigger_time = now
                            break
                    elif idx == 0:
                        self.thumb_frames = max(0, self.thumb_frames - 1)

            # 4. Two-Hand Comparison Mode
            if not in_cooldown and gesture_event is None and num_hands == 2:
                self.dual_frames += 1
                gesture_progress = max(
                    gesture_progress,
                    min(1.0, self.dual_frames / self.DUAL_REQUIRED),
                )
                if self.dual_frames >= self.DUAL_REQUIRED:
                    gesture_event = "DUAL_COMPARE"
                    self._reset_counters()
                    self.last_trigger_time = now
            else:
                self.dual_frames = max(0, self.dual_frames - 1)
        else:
            self._reset_counters()

        return annotated, gesture_event, gesture_progress, hand_data, centering_hint

    def _reset_counters(self):
        self.pinch_frames = 0
        self.palm_frames = 0
        self.dual_frames = 0
        self.thumb_frames = 0

    def close(self):
        if not self.use_tasks_api:
            self.hands.close()
        else:
            self.landmarker.close()