import json
import os
import cv2
import numpy as np
from PIL import Image

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class VisionAnalyzer:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.client = None
        if HAS_GENAI and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                print("[VisionAnalyzer] Gemini Vision-Language Model connected.")
            except Exception as e:
                print(f"[VisionAnalyzer] Gemini init failed, using OpenCV offline mode: {e}")
        else:
            print("[VisionAnalyzer] Running in Offline OpenCV HSV + Ripeness Mode (Set GEMINI_API_KEY for VLM mode).")

    def analyze_single_fruit(self, bgr_crop: np.ndarray) -> dict:
        """
        Returns a structured dictionary:
        {
            'fruit_name': str,
            'surface_description': str,
            'urgency': 'EAT_TODAY' | 'USE_SOON' | 'STORE_LONGER',
            'spoken_text': str
        }
        """
        if bgr_crop is None or bgr_crop.size == 0:
            return {
                "fruit_name": "Unknown",
                "surface_description": "No clear fruit region captured.",
                "urgency": "USE_SOON",
                "spoken_text": "Could not see the fruit clearly. Hold it near the center and pinch again.",
            }

        if self.client is not None:
            vlm_result = self._analyze_with_gemini(bgr_crop)
            if vlm_result is not None:
                return vlm_result

        return self._analyze_with_opencv_hsv(bgr_crop)

    def compare_two_fruits(self, left_crop: np.ndarray, right_crop: np.ndarray) -> str:
        """Compare two fruits held in Left and Right hands and return spoken recommendation."""
        left_info = self.analyze_single_fruit(left_crop)
        right_info = self.analyze_single_fruit(right_crop)

        priority_rank = {"EAT_TODAY": 1, "USE_SOON": 2, "STORE_LONGER": 3}
        l_rank = priority_rank.get(left_info["urgency"], 2)
        r_rank = priority_rank.get(right_info["urgency"], 2)

        if l_rank < r_rank:
            recommendation = "Use the fruit in your left hand first, and save the right hand fruit for later."
        elif r_rank < l_rank:
            recommendation = "Use the fruit in your right hand first, and save the left hand fruit for later."
        else:
            recommendation = "Both fruits have similar surface ripeness and can be used in either order."

        return (
            f"Left hand holds {left_info['fruit_name']} with {left_info['surface_description']}. "
            f"Right hand holds {right_info['fruit_name']} with {right_info['surface_description']}. "
            f"{recommendation}"
        )

    def _analyze_with_gemini(self, bgr_crop: np.ndarray):
        """Image-to-Text analysis using Gemini 2.5 Flash."""
        try:
            rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            prompt = (
                "You are FreshPick, an assistive Image-to-Text translator for a visually impaired user. "
                "Examine the fruit or produce held in the user's hand. "
                "Respond ONLY in valid JSON with these exact keys:\n"
                "{\n"
                '  "fruit_name": "name of fruit (e.g. Banana, Apple, Tomato, Orange)",\n'
                '  "surface_description": "short description of visible peel color, spots, or wrinkles",\n'
                '  "urgency": "EAT_TODAY or USE_SOON or STORE_LONGER",\n'
                '  "spoken_text": "2 clear spoken sentences describing the fruit, visible surface ripeness, '
                'whether to eat it first or keep it longer, and a brief reminder that camera only sees surface '
                'so gently check firmness."\n'
                "}"
            )

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[pil_img, prompt],
            )
            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`").replace("json", "", 1).strip()

            data = json.loads(raw_text)
            return {
                "fruit_name": data.get("fruit_name", "Fruit"),
                "surface_description": data.get("surface_description", "visible surface analyzed"),
                "urgency": data.get("urgency", "USE_SOON"),
                "spoken_text": data.get("spoken_text", "Fruit scanned."),
            }
        except Exception as e:
            print(f"[VisionAnalyzer] VLM fallback triggered: {e}")
            return None

    def _analyze_with_opencv_hsv(self, bgr_crop: np.ndarray) -> dict:
        """
        Offline HSV peel color + dark spot ratio analyzer.
        Estimates dominant fruit color and surface browning/spotting.
        """
        resized = cv2.resize(bgr_crop, (160, 160))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # Ignore very dark background or skin-tone neutral regions where possible
        total_pixels = resized.shape[0] * resized.shape[1]

        # Color masks in HSV
        yellow_mask = cv2.inRange(hsv, (18, 70, 70), (35, 255, 255))
        green_mask = cv2.inRange(hsv, (36, 50, 50), (85, 255, 255))
        red_mask1 = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (165, 80, 60), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        orange_mask = cv2.inRange(hsv, (11, 80, 70), (17, 255, 255))
        brown_dark_mask = cv2.inRange(hsv, (0, 30, 15), (25, 200, 95))

        yellow_ratio = np.count_nonzero(yellow_mask) / total_pixels
        green_ratio = np.count_nonzero(green_mask) / total_pixels
        red_ratio = np.count_nonzero(red_mask) / total_pixels
        orange_ratio = np.count_nonzero(orange_mask) / total_pixels
        spot_ratio = np.count_nonzero(brown_dark_mask) / total_pixels

        color_scores = {
            "Banana or Yellow Fruit": yellow_ratio,
            "Green Apple or Unripe Fruit": green_ratio,
            "Apple or Tomato": red_ratio,
            "Orange or Citrus": orange_ratio,
        }
        best_fruit = max(color_scores, key=color_scores.get)

        # Determine ripeness urgency based on dark spots and green/yellow balance
        if spot_ratio > 0.18:
            urgency = "EAT_TODAY"
            surface_desc = "noticeable dark surface spots and mature peel tone"
            advice = "Use this first, ideally today. Gently check firmness since cameras only inspect the outside."
        elif green_ratio > 0.22 and green_ratio > yellow_ratio:
            urgency = "STORE_LONGER"
            surface_desc = "mostly firm green surface with minimal spotting"
            advice = "This is still underripe and can stay for several days."
        else:
            urgency = "USE_SOON"
            surface_desc = "even, ripe surface color with light natural blemishes"
            advice = "Ready to eat now or within two to three days. Check smell and firmness before cutting."

        spoken_text = (
            f"Detected {best_fruit} showing {surface_desc}. {advice}"
        )

        return {
            "fruit_name": best_fruit,
            "surface_description": surface_desc,
            "urgency": urgency,
            "spoken_text": spoken_text,
        }