import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_engine import AudioEngine
from config import (
    CAMERA_INDEX,
    FLIP_CAMERA,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    FULLSCREEN_BY_DEFAULT,
    WINDOW_NAME,
)
from gesture_controller import GestureController
from hud import HUD
from pantry_memory import PantryMemory
from state_machine import InspectionStateMachine
from vision_analyzer import VisionAnalyzer


def main():
    audio = AudioEngine(rate=175)
    gesture_ctrl = GestureController()
    vision = VisionAnalyzer()
    pantry = PantryMemory()
    hud = HUD()
    fsm = InspectionStateMachine(audio=audio, vision=vision, pantry=pantry)

    # Use DirectShow on Windows for faster high-res camera initialization
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        audio.speak("Camera could not be opened.")
        return

    # Configure Full-Screen OpenCV Window
    is_fullscreen = FULLSCREEN_BY_DEFAULT
    is_flipped = FLIP_CAMERA

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if is_fullscreen:
        cv2.setWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

    audio.speak(
        "FreshPick full screen ready. Hold a fruit inside the center box for a 10 second 360 freshness scan."
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Normal camera by default; press 'F' to toggle mirror if needed
            if is_flipped:
                frame = cv2.flip(frame, 1)

            detection = gesture_ctrl.process_frame(frame)
            fsm.update(detection)

            display_frame = hud.render(
                frame=detection["annotated_frame"],
                detection=detection,
                state_name=fsm.state,
                elapsed_10s=fsm.get_10s_elapsed(),
                rotation_deg=fsm.rotation_deg,
                caption=fsm.caption,
                pantry_items=pantry.get_sorted_items(),
                is_flipped=is_flipped,
            )

            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # Q or ESC to quit
                break
            elif key == ord("f"):  # Toggle Normal vs Mirror camera live
                is_flipped = not is_flipped
            elif key == ord(" "):  # SPACEBAR: Instant scan of center zone
                h, w, _ = frame.shape
                center_crop = frame[int(h * 0.2):int(h * 0.8), int(w * 0.28):int(w * 0.72)].copy()
                target_crop = detection["primary_crop"] if detection["primary_crop"] is not None else center_crop
                audio.play_lock_chime()
                fsm.trigger_instant_scan(target_crop)
            elif key == ord("r"):  # R: Read out Use-First to Use-Last list
                recap = pantry.get_spoken_recap()
                fsm.caption = recap
                audio.speak(recap, interrupt=True)
            elif key == ord("w"):  # W: Toggle Fullscreen / Windowed
                is_fullscreen = not is_fullscreen
                cv2.setWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL,
                )

    finally:
        cap.release()
        gesture_ctrl.close()
        audio.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()