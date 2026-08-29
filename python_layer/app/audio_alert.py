import queue
import threading
import time
from typing import Optional

class AudioAlertEngine:
    """Non-blocking priority-queued Text-to-Speech alert engine."""

    DANGER = 0
    WARNING = 1
    INFO = 2
    COOLDOWN = 5.0  # Seconds between repeat audio triggers for identical alerts

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._queue = queue.PriorityQueue()
        self._cooldowns = {}
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if self.enabled:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def process(self, advisory, context):
        if not self.enabled:
            return

        if advisory.level == "DANGER":
            self.speak(advisory.message, priority=self.DANGER, key="DANGER_PRIMARY")
        elif advisory.level == "WARNING":
            self.speak(advisory.message, priority=self.WARNING, key=advisory.message)
        elif context.hazard_level == "critical":
            self.speak("Critical Hazard", priority=self.DANGER, key="HAZARD_CRITICAL")

    def speak(self, text: str, priority: int = WARNING, key: Optional[str] = None):
        if not self.enabled:
            return

        now = time.time()
        dedup_key = key or text
        last_spoken = self._cooldowns.get(dedup_key, 0.0)

        # Allow immediate interrupt for DANGER, enforce cooldown on others
        if priority != self.DANGER and (now - last_spoken) < self.COOLDOWN:
            return

        self._cooldowns[dedup_key] = now
        # Tuple format for PriorityQueue: (priority, insertion_timestamp, text)
        self._queue.put((priority, now, text))

    def _worker(self):
        # Initialize pyttsx3 within the worker thread for safe native engine binding
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 185)
        except Exception:
            return

        while not self._stop_event.is_set():
            try:
                priority, _, text = self._queue.get(timeout=0.1)
                engine.say(text)
                engine.runAndWait()
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)