from __future__ import annotations

import io
import logging
import wave

import httpx
import numpy as np

log = logging.getLogger("LiveTranslate.Voxtral")


class VoxtralASREngine:
    """Speech-to-text via an OpenAI-compatible audio transcription endpoint.

    Endpoint:
    - base URL: http://127.0.0.1:8000
    - transcriptions: /v1/audio/transcriptions
    """

    def __init__(
        self,
        api_base: str = "http://127.0.0.1:8000",
        model: str = "Voxtral-Mini-4B-Realtime-2602-GGUF",
        timeout: float = 60.0,
        device: str = "cpu",
    ):
        self._api_base = api_base.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client = httpx.Client(base_url=self._api_base, timeout=timeout, trust_env=False)
        self.language = None  # None = auto detect
        log.info(f"Voxtral loaded via {self._api_base}/v1/audio/transcriptions ({device})")

    def set_language(self, language: str):
        old = self.language
        self.language = language if language != "auto" else None
        log.info(f"Voxtral language: {old} -> {self.language}")

    def to_device(self, device: str):
        # Remote/OpenAI-compatible ASR service; nothing to migrate locally.
        return False

    def unload(self):
        if hasattr(self, "_client") and self._client is not None:
            self._client.close()
            self._client = None

    def transcribe(self, audio: np.ndarray) -> dict | None:
        if self._client is None:
            return None

        wav_bytes = self._to_wav_bytes(audio)
        data = {
            "model": self._model,
            "response_format": "json",
        }
        if self.language:
            data["language"] = self.language

        files = {
            "file": ("audio.wav", wav_bytes, "audio/wav"),
        }

        resp = self._client.post("/v1/audio/transcriptions", data=data, files=files)
        resp.raise_for_status()
        payload = resp.json()

        text = (payload.get("text") or "").strip()
        if not text:
            return None

        detected_lang = payload.get("language") or self.language or "auto"
        log.debug(f"Voxtral result: {text}")
        return {
            "text": text,
            "language": detected_lang,
            "language_name": detected_lang,
        }

    @staticmethod
    def _to_wav_bytes(audio: np.ndarray) -> bytes:
        clipped = np.clip(audio, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()
