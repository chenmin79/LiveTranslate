from __future__ import annotations

from dataclasses import dataclass, field

from pipeline.segment_assembler import SpeechSegment
from services.translation_service import TranslationResult


@dataclass(slots=True)
class BilingualSubtitle:
    """Display-ready bilingual subtitle payload."""

    segment_id: int
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    timestamp: str = ""
    is_preview: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


class BilingualFormatter:
    """Combines original ASR text and translated text into one UI payload.

    This is the future seam between pipeline text production and a dedicated
    bilingual floating subtitle presentation layer.
    """

    def format(
        self,
        segment: SpeechSegment,
        translation: TranslationResult,
    ) -> BilingualSubtitle:
        return BilingualSubtitle(
            segment_id=segment.segment_id,
            original_text=segment.text,
            translated_text=translation.text,
            source_language=segment.source_language,
            target_language=translation.target_language,
            timestamp=segment.timestamp,
            is_preview=segment.is_preview,
            metadata={
                **segment.metadata,
                **translation.metadata,
            },
        )

    def format_preview(
        self,
        segment: SpeechSegment,
        partial_translation: str,
        *,
        target_language: str,
    ) -> BilingualSubtitle:
        return BilingualSubtitle(
            segment_id=segment.segment_id,
            original_text=segment.text,
            translated_text=partial_translation,
            source_language=segment.source_language,
            target_language=target_language,
            timestamp=segment.timestamp,
            is_preview=True,
            metadata=dict(segment.metadata),
        )
