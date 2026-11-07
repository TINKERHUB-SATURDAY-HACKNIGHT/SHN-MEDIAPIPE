import threading
import time
import numpy as np
from config import (
    COOLDOWN_SEC,
    HOMING_HOLD_SEC,
    READOUT_DURATION_SEC,
    ROTATION_DURATION_SEC,
)


class InspectionStateMachine:
    STATE_IDLE = "IDLE"
    STATE_HOMING = "HOMING (0-1.2s)"
    STATE_ROTATING = "ROTATE 360 (1.2-4.4s)"
    STATE_ANALYZING = "ANALYZING SURFACE"
    STATE_READOUT = "VOICE INSTRUCTION (4.4-10s)"

    def __init__(self, audio, vision, pantry):
        self.audio = audio
        self.vision = vision
        self.pantry = pantry

        self.state = self.STATE_IDLE
        self.cycle_start_time = 0.0
        self.state_start_time = 0.0
        self.last_complete_time = 0.0

        self.front_crop = None
        self.back_crop = None
        self.base_normal = None
        self.rotation_deg = 0.0

        self.caption = "Hold any fruit in the center box (or press SPACE / Pinch) to start the 10s 360° scan."

    def get_10s_elapsed(self) -> float:
        if self.state == self.STATE_IDLE:
            return 0.0
        return min(10.0, time.time() - self.cycle_start_time)

    def trigger_instant_scan(self, crop_img):
        """Immediate scan when user pinches or presses SPACE."""
        if self.state in (self.STATE_ANALYZING,):
            return
        self.cycle_start_time = time.time()
        self.front_crop = crop_img
        self.back_crop = crop_img
        self._start_analysis()

    def update(self, detection: dict):
        now = time.time()
        in_cooldown = (now - self.last_complete_time) < COOLDOWN_SEC

        # Handle Open Palm Recap at any idle/readout time
        if detection["instant_gesture"] == "PALM_RECAP" and self.state != self.STATE_ANALYZING:
            recap = self.pantry.get_spoken_recap()
            self.caption = recap
            self.audio.play_beep(1000, 100)
            self.audio.speak(recap, interrupt=True)
            self.last_complete_time = now
            return

        # Handle Instant Pinch Scan
        if detection["instant_gesture"] == "PINCH_SCAN" and detection["primary_crop"] is not None:
            self.audio.play_lock_chime()
            self.trigger_instant_scan(detection["primary_crop"])
            return

        if self.state == self.STATE_IDLE:
            if in_cooldown:
                return
            if detection["target_present"]:
                if detection["centered"]:
                    self.state = self.STATE_HOMING
                    self.cycle_start_time = now
                    self.state_start_time = now
                    self.audio.play_beep(750, 80)
                    self.caption = "Fruit detected in scan zone... Hold steady for 1 second to lock."
                elif detection["centering_hint"]:
                    self.audio.speak_hint(detection["centering_hint"])

        elif self.state == self.STATE_HOMING:
            if not detection["target_present"]:
                self.state = self.STATE_IDLE
                self.caption = "Hold any fruit in the center scan box to begin."
                return

            if detection["primary_crop"] is not None:
                self.front_crop = detection["primary_crop"]
            if detection["palm_normal"] is not None:
                self.base_normal = detection["palm_normal"]

            if (now - self.state_start_time) >= HOMING_HOLD_SEC:
                # Front view captured! Transition to 360 Rotation phase
                self.state = self.STATE_ROTATING
                self.state_start_time = now
                self.rotation_deg = 0.0
                self.audio.play_lock_chime()
                self.caption = "Front side captured! Slowly turn your wrist to show all sides of the fruit..."
                self.audio.speak("Front captured. Slowly turn the fruit.", interrupt=True)

        elif self.state == self.STATE_ROTATING:
            if detection["primary_crop"] is not None:
                self.back_crop = detection["primary_crop"]

            # Compute 3D wrist rotation angle if hand normal is visible
            if self.base_normal is not None and detection["palm_normal"] is not None:
                dot = float(np.clip(np.dot(self.base_normal, detection["palm_normal"]), -1.0, 1.0))
                angle = float(np.degrees(np.arccos(dot)))
                self.rotation_deg = max(self.rotation_deg, angle)

            rot_elapsed = now - self.state_start_time
            # Complete rotation when 3.2s elapses OR user twists wrist > 110 degrees after 1.5s
            if rot_elapsed >= ROTATION_DURATION_SEC or (self.rotation_deg >= 110.0 and rot_elapsed >= 1.5):
                if self.back_crop is None:
                    self.back_crop = self.front_crop
                self.audio.play_complete_chime()
                self._start_analysis()

        elif self.state == self.STATE_READOUT:
            if (now - self.state_start_time) >= READOUT_DURATION_SEC:
                self.state = self.STATE_IDLE
                self.last_complete_time = now
                self.caption = "10s Scan Complete! Show next fruit, or show Open Palm (or press R) for full priority list."

    def _start_analysis(self):
        self.state = self.STATE_ANALYZING
        self.state_start_time = time.time()
        self.caption = "Analyzing 360° surface ripeness and updating Use-First / Use-Last order..."

        def _worker(front_img, back_img):
            result = self.vision.analyze_360_fruit(front_img, back_img)
            self.pantry.add_item(result)
            order_note = self.pantry.get_comparison_summary()
            full_speech = f"{result['spoken_text']} {order_note}"

            self.caption = full_speech
            self.state = self.STATE_READOUT
            self.state_start_time = time.time()
            self.audio.speak(full_speech, interrupt=True)

        threading.Thread(
            target=_worker,
            args=(self.front_crop, self.back_crop),
            daemon=True,
        ).start()