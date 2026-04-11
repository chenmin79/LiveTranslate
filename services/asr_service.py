from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class ASRRequest:
    """Input payload for a future ASR service abstraction."""

    audio: np.ndarray
    use_word_timestamps: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ASRResult:
    """Normalized ASR output shared by all engine backends."""

    text: str
    language: str
    language_name: str
    words: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


class SupportsTranscribe(Protocol):
    """Minimal protocol implemented by the existing ASR engine classes."""

    def transcribe(self, audio: np.ndarray, **kwargs) -> dict | None:
        ...


class ASRService:
    """Thin wrapper around the existing ASR engine contract.

    This does not replace the current runtime path yet. Its job is to provide a
    stable boundary so `main.py` can later depend on one service interface
    instead of branching on engine details.
    """

    def __init__(self, engine: SupportsTranscribe):
        self._engine = engine

    def transcribe(self, request: ASRRequest) -> ASRResult | None:
        kwargs = {}
        if request.use_word_timestamps:
            kwargs["word_timestamps"] = True

        try:
            raw = self._engine.transcribe(request.audio, **kwargs)
        except TypeError:
            raw = self._engine.transcribe(request.audio)
        if not raw:
            return None

        return ASRResult(
            text=(raw.get("text") or "").strip(),
            language=raw.get("language", "auto"),
            language_name=raw.get("language_name", raw.get("language", "auto")),
            words=list(raw.get("words") or []),
            metadata=dict(request.metadata),
        )
