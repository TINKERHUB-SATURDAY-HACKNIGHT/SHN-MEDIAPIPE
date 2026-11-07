import os
import sys
import threading
import cv2

# Ensure local sibling modules resolve regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_engine import AudioEngine
from gesture_controller import GestureController
from hud import HUD
from pantry_memory import PantryMemory
from vision_analyzer import VisionAnalyzer


def main():
    audio = AudioEngine(rate=175)
    gesture_ctrl = GestureController()
    vision = VisionAnalyzer()
    pantry = PantryMemory()
    hud = HUD()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        audio.speak("Camera could not be opened. Please check your webcam connection.")
        return

    state = {
        "caption": "Hold a fruit in front of the camera and pinch thumb + index finger to scan.",
        "busy": False,
    }

    audio.speak(
        "FreshPick ready. Hold a fruit near the center and pinch to hear its visual description and freshness priority."
    )

    def run_single_scan(crop_img):
        state["busy"] = True
        state["caption"] = "Scanning fruit surface and translating image to speech..."
        try:
            result = vision.analyze_single_fruit(crop_img)
            pantry.add_item(result)
            state["caption"] = result["spoken_text"]
            audio.speak(result["spoken_text"], interrupt=True)
        finally:
            state["busy"] = False

    def run_dual_comparison(left_crop, right_crop):
        state["busy"] = True
        state["caption"] = "Comparing fruits in your left and right hands..."
        try:
            comparison_speech = vision.compare_two_fruits(left_crop, right_crop)
            state["caption"] = comparison_speech
            audio.speak(comparison_speech, interrupt=True)
        finally:
            state["busy"] = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror view for intuitive hand interaction
            frame = cv2.flip(frame, 1)

            (
                annotated_frame,
                gesture_event,
                gesture_progress,
                hand_data,
                centering_hint,
            ) = gesture_ctrl.process_frame(frame)

            # Give gentle audio centering hints if hand is near edge of camera
            if centering_hint and not state["busy"]:
                audio.speak_centering_hint(centering_hint)

            # Handle Gesture Events
            if gesture_event and not state["busy"]:
                audio.play_scan_trigger_tone()

                if gesture_event == "PINCH_SCAN" and hand_data:
                    # If 2 hands are visible (1 holding fruit, 1 pinching), scan the non-pinching hand crop or largest crop
                    target_crop = max(
                        hand_data,
                        key=lambda h: (h["bbox"][2] - h["bbox"][0]) * (h["bbox"][3] - h["bbox"][1]),
                    )["crop"]
                    threading.Thread(
                        target=run_single_scan, args=(target_crop,), daemon=True
                    ).start()

                elif gesture_event == "DUAL_COMPARE" and len(hand_data) >= 2:
                    # Sort hands left-to-right on screen
                    sorted_hands = sorted(hand_data, key=lambda h: h["center"][0])
                    left_crop = sorted_hands[0]["crop"]
                    right_crop = sorted_hands[1]["crop"]
                    threading.Thread(
                        target=run_dual_comparison,
                        args=(left_crop, right_crop),
                        daemon=True,
                    ).start()

                elif gesture_event == "PALM_RECAP":
                    recap_text = pantry.get_spoken_recap()
                    state["caption"] = recap_text
                    audio.speak(recap_text, interrupt=True)

                elif gesture_event == "THUMB_SAFETY":
                    safety_note = (
                        "Safety reminder: Cameras only see outer peel color and spots. "
                        "Always press gently for soft spots and smell near the stem to check for internal spoilage."
                    )
                    state["caption"] = safety_note
                    audio.speak(safety_note, interrupt=True)

            # Render accessibility HUD
            display_frame = hud.render(
                frame=annotated_frame,
                hand_data=hand_data,
                gesture_progress=gesture_progress,
                last_caption=state["caption"],
                pantry_items=pantry.get_sorted_items(),
            )

            cv2.imshow("FreshPick - Beyond Gesture (SHN #01)", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("r"):
                # Keyboard shortcut backup for demoing Pantry Recap
                recap_text = pantry.get_spoken_recap()
                state["caption"] = recap_text
                audio.speak(recap_text, interrupt=True)

    finally:
        cap.release()
        gesture_ctrl.close()
        audio.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()