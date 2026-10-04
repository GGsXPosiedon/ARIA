"""Optional offline microphone recognition and Windows speech output."""

import json
import os
from pathlib import Path
from queue import Empty, Queue
import time


class VoiceError(RuntimeError):
    """Offline speech input or output is unavailable."""


class LocalVoice:
    def __init__(self, model_path: Path):
        if not model_path.is_dir():
            raise VoiceError(f"Vosk speech model not found at {model_path}. See README voice setup.")
        try:
            import sounddevice as sd
            import vosk
            import pyttsx3
        except ImportError as exc:
            raise VoiceError("Voice packages are missing. Run: python -m pip install -r requirements-voice.txt") from exc
        try:
            self.model = vosk.Model(str(model_path))
            self.speaker = pyttsx3.init()
        except Exception as exc:
            raise VoiceError(f"Cannot start offline speech: {exc}") from exc
        self.sd = sd
        self.vosk = vosk

    def listen(self, timeout_seconds: int = 20) -> str:
        audio = Queue()

        def callback(data, _frames, _time, status):
            if status:
                audio.put(status)
            audio.put(bytes(data))

        try:
            device = self.sd.query_devices(kind="input")
            rate = int(device["default_samplerate"])
            recognizer = self.vosk.KaldiRecognizer(self.model, rate)
            deadline = time.monotonic() + timeout_seconds
            with self.sd.RawInputStream(samplerate=rate, blocksize=8000, dtype="int16",
                                        channels=1, callback=callback):
                while time.monotonic() < deadline:
                    try:
                        chunk = audio.get(timeout=min(0.5, max(0.01, deadline - time.monotonic())))
                    except Empty:
                        continue
                    if not isinstance(chunk, bytes):
                        raise VoiceError(f"Microphone error: {chunk}")
                    if recognizer.AcceptWaveform(chunk):
                        text = json.loads(recognizer.Result()).get("text", "").strip()
                        if text:
                            return text
            text = json.loads(recognizer.FinalResult()).get("text", "").strip()
            if text:
                return text
            raise VoiceError("No speech was recognized. Please try again.")
        except VoiceError:
            raise
        except Exception as exc:
            raise VoiceError(f"Microphone recognition failed: {exc}") from exc

    def say(self, text: str) -> None:
        try:
            self.speaker.say(text)
            self.speaker.runAndWait()
        except Exception as exc:
            raise VoiceError(f"Could not speak the reply: {exc}") from exc


def model_path() -> Path:
    return Path(os.getenv("ARIA_VOSK_MODEL_PATH", "models/vosk-model-small-en-us-0.15")).expanduser()
