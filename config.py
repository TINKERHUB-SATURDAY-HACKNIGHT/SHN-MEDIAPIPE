import os

# Load .env file automatically if present
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(ENV_PATH):
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# Gemini API Configuration (Updated to gemini-3.8-flash)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip()

# Camera & Window Settings
WINDOW_NAME = "FreshPick - 360 Assistive Fruit Freshness Tracker"
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FULLSCREEN_BY_DEFAULT = True
FLIP_CAMERA = False  # False = Normal camera view | True = Mirror view (Press 'F' to toggle live)

# MediaPipe Detection Thresholds (Lowered so holding a fruit doesn't break tracking)
MP_DETECTION_CONFIDENCE = 0.35
MP_PRESENCE_CONFIDENCE = 0.35
MP_TRACKING_CONFIDENCE = 0.35
ROI_PADDING = 0.16

# 10-Second Guided Inspection Timings (in seconds)
HOMING_HOLD_SEC = 1.2       # 0.0s - 1.2s: Lock onto fruit in scan zone
ROTATION_DURATION_SEC = 3.2 # 1.2s - 4.4s: Turn fruit 360 degrees (Front + Back capture)
READOUT_DURATION_SEC = 5.6  # 4.4s - 10.0s: Voice instructions & "Use First vs Last" ranking
COOLDOWN_SEC = 2.5          # Pause before next automatic scan