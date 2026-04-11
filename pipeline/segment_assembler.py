from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(slots=True)
class SpeechSegment:
    """A normalized speech segment ready for downstream ASR/translation work."""

    segment_id: int
    text: str
    source_language: str
    asr_ms: float = 0.0
    timestamp: str = ""
    is_preview: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SegmentBatch:
    """A small batch container for future fan-out or grouped UI updates."""

    segments: list[SpeechSegment] = field(default_factory=list)

    def append(self, segment: SpeechSegment) -> None:
        self.segments.append(segment)

    def extend(self, segments: Iterable[SpeechSegment]) -> None:
        self.segments.extend(segments)

    def latest(self) -> SpeechSegment | None:
        return self.segments[-1] if self.segments else None

    def is_empty(self) -> bool:
        return not self.segments


class SegmentAssembler:
    """Builds stable segment objects from raw ASR output.

    This class is intentionally small for now. The existing application still
    assembles text inside `main.py`. This module provides the future seam for:

    - preview/final segment normalization
    - segment metadata enrichment
    - consistent IDs/timestamps for multiple display targets
    """

    def build_segment(
        self,
        *,
        segment_id: int,
        text: str,
        source_language: str,
        asr_ms: float = 0.0,
        timestamp: str = "",
        is_preview: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> SpeechSegment:
        return SpeechSegment(
            segment_id=segment_id,
            text=text.strip(),
            source_language=source_language,
            asr_ms=asr_ms,
            timestamp=timestamp,
            is_preview=is_preview,
            metadata=dict(metadata or {}),
        )

    def build_batch(self, segments: Iterable[SpeechSegment]) -> SegmentBatch:
        batch = SegmentBatch()
        batch.extend(segments)
        return batch
