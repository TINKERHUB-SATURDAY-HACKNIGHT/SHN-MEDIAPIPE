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
        self.hint_cooldown = 3.5

        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._tts_worker, daemon=True)
        self._worker.start()

    def _tts_worker(self):
        while not self._stop_event.is_set():
            try:
                text = self.speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:
                break

            self.is_speaking = True
            print(f"[FreshPick Voice]: {text}")
            if HAS_TTS:
                try:
                    # Re-initialize per utterance on Windows COM threads for reliability
                    engine = pyttsx3.init()
                    engine.setProperty("rate", self.rate)
                    engine.setProperty("volume", 1.0)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    print(f"[AudioEngine] TTS Warning: {e}")
            self.is_speaking = False
            self.speech_queue.task_done()

    def speak(self, text: str, interrupt: bool = False):
        if not text:
            return
        if interrupt:
            with self.speech_queue.mutex:
                self.speech_queue.queue.clear()
        self.speech_queue.put(text)

    def speak_hint(self, hint: str):
        now = time.time()
        if hint and not self.is_speaking and (now - self.last_hint_time) > self.hint_cooldown:
            self.last_hint_time = now
            self.speak(hint)

    def play_beep(self, freq: int = 880, duration_ms: int = 110):
        if not HAS_WINSOUND:
            return
        threading.Thread(
            target=lambda: winsound.Beep(freq, duration_ms),
            daemon=True,
        ).start()

    def play_lock_chime(self):
        def _chime():
            if HAS_WINSOUND:
                winsound.Beep(880, 80)
                winsound.Beep(1320, 130)
        threading.Thread(target=_chime, daemon=True).start()

    def play_complete_chime(self):
        def _chime():
            if HAS_WINSOUND:
                winsound.Beep(1046, 90)
                winsound.Beep(1318, 90)
                winsound.Beep(1567, 160)
        threading.Thread(target=_chime, daemon=True).start()

    def shutdown(self):
        self._stop_event.set()
        self.speech_queue.put(None)