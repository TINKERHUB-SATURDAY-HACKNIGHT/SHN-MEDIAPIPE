import json
import cv2
import numpy as np
from PIL import Image
import config

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class VisionAnalyzer:
    def __init__(self, use_gemini: bool = True):
        self.api_key = getattr(config, "GEMINI_API_KEY", "")
        self.model_name = getattr(config, "GEMINI_MODEL", "gemini-3.8-flash")
        self.use_gemini = use_gemini and bool(self.api_key) and self.api_key != "your_api_key_here"
        self.client = None

        if HAS_GENAI and self.use_gemini:
            try:
                self.client = genai.Client(api_key=self.api_key)
                print(f"[VisionAnalyzer] Connected to {self.model_name}.")
            except Exception as e:
                print(f"[VisionAnalyzer] VLM init warning, using offline HSV analyzer: {e}")
                self.client = None
        else:
            print("[VisionAnalyzer] Using fast Offline Multi-Fruit HSV + Spot Analyzer.")

    def analyze_360_fruit(self, front_crop: np.ndarray, back_crop: np.ndarray = None) -> dict:
        if front_crop is None or front_crop.size == 0:
            return {
                "fruit_name": "Unknown Fruit",
                "surface_description": "Fruit was outside the scan window",
                "urgency": "USE_SOON",
                "shelf_life_days": "2-3 days",
                "spoken_text": "Could not capture the fruit clearly. Please hold the fruit inside the center box.",
            }

        if back_crop is None or back_crop.size == 0:
            back_crop = front_crop

        if self.client is not None:
            vlm_res = self._analyze_with_gemini(front_crop, back_crop)
            if vlm_res is not None:
                return vlm_res

        return self._analyze_offline_360(front_crop, back_crop)

    # Alias for backward compatibility
    def analyze_single_fruit(self, crop: np.ndarray) -> dict:
        return self.analyze_360_fruit(crop, crop)

    def _analyze_with_gemini(self, front_crop: np.ndarray, back_crop: np.ndarray):
        try:
            img_front = Image.fromarray(cv2.cvtColor(front_crop, cv2.COLOR_BGR2RGB))
            img_back = Image.fromarray(cv2.cvtColor(back_crop, cv2.COLOR_BGR2RGB))

            prompt = (
                "You are FreshPick, an assistive voice system for visually impaired users. "
                "You are given two images of a fruit held in a user's hand: Front view and Back view. "
                "Respond ONLY in valid JSON with these exact keys:\n"
                "{\n"
                '  "fruit_name": "specific fruit name (e.g. Banana, Mango, Apple, Orange, Tomato, Papaya)",\n'
                '  "surface_description": "concise description of peel color and any spots across both sides",\n'
                '  "urgency": "EAT_TODAY or USE_SOON or STORE_LONGER",\n'
                '  "shelf_life_days": "estimated days (e.g. Eat today, 2-3 days, 4-6 days)",\n'
                '  "spoken_text": "Under 22 words: Name the fruit, describe front and back ripeness, state if it should be used first today or stored longer, and remind to gently press for firmness."\n'
                "}"
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[img_front, img_back, prompt],
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.strip("`").replace("json", "", 1).strip()
            data = json.loads(raw)
            return {
                "fruit_name": data.get("fruit_name", "Fruit"),
                "surface_description": data.get("surface_description", "360 surface inspected"),
                "urgency": data.get("urgency", "USE_SOON"),
                "shelf_life_days": data.get("shelf_life_days", "2-3 days"),
                "spoken_text": data.get("spoken_text", "Fruit inspected."),
            }
        except Exception as e:
            print(f"[VisionAnalyzer] VLM fallback to OpenCV HSV: {e}")
            return None

    def _analyze_offline_360(self, front_crop: np.ndarray, back_crop: np.ndarray) -> dict:
        """Combines Front and Back crops in HSV space to detect fruit type, ripeness, and dark spots."""
        f_img = cv2.resize(front_crop, (160, 160))
        b_img = cv2.resize(back_crop, (160, 160))
        combined = np.hstack([f_img, b_img])
        hsv = cv2.cvtColor(combined, cv2.COLOR_BGR2HSV)
        total_px = float(combined.shape[0] * combined.shape[1])

        yellow = np.count_nonzero(cv2.inRange(hsv, (18, 65, 65), (35, 255, 255))) / total_px
        orange = np.count_nonzero(cv2.inRange(hsv, (10, 80, 70), (18, 255, 255))) / total_px
        red1 = cv2.inRange(hsv, (0, 75, 55), (10, 255, 255))
        red2 = cv2.inRange(hsv, (160, 75, 55), (180, 255, 255))
        red = np.count_nonzero(cv2.bitwise_or(red1, red2)) / total_px
        green = np.count_nonzero(cv2.inRange(hsv, (36, 45, 45), (88, 255, 255))) / total_px
        dark_spots = np.count_nonzero(cv2.inRange(hsv, (0, 35, 15), (28, 210, 85))) / total_px

        scores = {
            "Banana / Mango": yellow,
            "Orange / Papaya": orange,
            "Apple / Tomato": red,
            "Green Fruit / Guava": green,
        }
        fruit_name = max(scores, key=scores.get)

        if dark_spots > 0.16:
            urgency = "EAT_TODAY"
            shelf_life = "Eat Today (Use First)"
            surface_desc = "ripe peel with dark spots detected on surface"
            order_msg = "Use this first today."
        elif green > 0.20 and green >= max(yellow, red, orange):
            urgency = "STORE_LONGER"
            shelf_life = "4 to 6 Days (Use Last)"
            surface_desc = "firm green unripe surface around the fruit"
            order_msg = "Unripe; keep this to use last."
        else:
            urgency = "USE_SOON"
            shelf_life = "2 to 3 Days"
            surface_desc = "even ripe color with clean surface"
            order_msg = "Ready to eat within two to three days."

        spoken_text = (
            f"360 scan complete: {fruit_name} with {surface_desc}. "
            f"{order_msg} Gently press to verify firmness."
        )

        return {
            "fruit_name": fruit_name,
            "surface_description": surface_desc,
            "urgency": urgency,
            "shelf_life_days": shelf_life,
            "spoken_text": spoken_text,
        }