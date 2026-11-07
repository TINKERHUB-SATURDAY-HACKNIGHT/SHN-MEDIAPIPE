import queue
import threading
import time

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False


class AudioEngine:
    def __init__(self, rate: int = 175):
        self.speech_queue = queue.Queue()
        self.rate = rate
        self.is_speaking = False
        self.last_hint_time = 0.0
        self.hint_cooldown = 4.0  # Seconds between centering guidance hints

        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._tts_worker, daemon=True)
        self._worker.start()

    def _tts_worker(self):
        """Dedicated thread for Windows SAPI5 pyttsx3 to prevent webcam freezing."""
        engine = None
        if HAS_TTS:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", 1.0)
            except Exception as e:
                print(f"[AudioEngine] TTS Init Warning: {e}")
                engine = None

        while not self._stop_event.is_set():
            try:
                text = self.speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:
                break

            self.is_speaking = True
            print(f"[FreshPick Voice]: {text}")
            if engine is not None:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print(f"[AudioEngine] TTS Error: {e}")
            self.is_speaking = False
            self.speech_queue.task_done()

    def speak(self, text: str, interrupt: bool = False):
        """Queue spoken feedback without blocking the OpenCV frame loop."""
        if not text:
            return
        if interrupt:
            with self.speech_queue.mutex:
                self.speech_queue.queue.clear()
        self.speech_queue.put(text)

    def speak_centering_hint(self, hint: str):
        """Speak spatial camera alignment hints with a cooldown to avoid spam."""
        now = time.time()
        if hint and not self.is_speaking and (now - self.last_hint_time) > self.hint_cooldown:
            self.last_hint_time = now
            self.speak(hint)

    def play_beep(self, freq: int = 880, duration_ms: int = 120):
        """Play non-blocking audio sonar cues on Windows."""
        if not HAS_WINSOUND:
            return
        threading.Thread(
            target=lambda: winsound.Beep(freq, duration_ms),
            daemon=True
        ).start()

    def play_scan_trigger_tone(self):
        """Double high tone when a gesture locks in."""
        def _tone():
            if HAS_WINSOUND:
                winsound.Beep(950, 90)
                winsound.Beep(1300, 140)
        threading.Thread(target=_tone, daemon=True).start()

    def shutdown(self):
        self._stop_event.set()
        self.speech_queue.put(None)