from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass(slots=True)
class TranslationRequest:
    """Input payload for future translation orchestration."""

    text: str
    source_language: str
    target_language: str
    is_preview: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class TranslationProgress:
    """Streaming translation progress payload."""

    partial_text: str
    done: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class TranslationResult:
    """Final translation output."""

    text: str
    source_language: str
    target_language: str
    metadata: dict[str, object] = field(default_factory=dict)


class TranslationService:
    """Thin adapter over the existing `Translator` class.

    For now this only formalizes request/result objects and pass-through
    streaming. Later it can own:

    - primary vs extra target fan-out
    - retries and timeout policy
    - preview/final translation coordination
    """

    def __init__(self, translator):
        self._translator = translator

    def translate(self, request: TranslationRequest) -> TranslationResult:
        text = self._translator.translate(
            request.text,
            request.source_language,
        )
        return TranslationResult(
            text=text,
            source_language=request.source_language,
            target_language=request.target_language,
            metadata=dict(request.metadata),
        )

    def translate_iter(self, request: TranslationRequest) -> Iterator[TranslationProgress]:
        for partial in self._translator.translate_iter(
            request.text,
            request.source_language,
        ):
            yield TranslationProgress(
                partial_text=partial,
                done=False,
                metadata=dict(request.metadata),
            )

    def with_target_language(self, target_language: str) -> "TranslationService":
        translator = self._translator.with_target_language(target_language)
        return TranslationService(translator)
