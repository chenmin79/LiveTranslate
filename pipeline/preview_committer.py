from __future__ import annotations

from dataclasses import dataclass, field

from pipeline.segment_assembler import SpeechSegment


@dataclass(slots=True)
class CommitDecision:
    """Result of deciding whether a preview should become a committed segment."""

    should_commit: bool
    committed_text: str = ""
    remainder_text: str = ""
    reason: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


class PreviewCommitter:
    """Owns preview-to-final text commit decisions.

    Today, interim commit logic is spread across `main.py`. This skeleton gives
    us a dedicated place to move that logic later without changing interfaces
    again.
    """

    def decide(
        self,
        preview_text: str,
        *,
        source_language: str,
        committed_tail: str = "",
    ) -> CommitDecision:
        text = preview_text.strip()
        if not text:
            return CommitDecision(
                should_commit=False,
                reason="empty_preview",
            )

        return CommitDecision(
            should_commit=True,
            committed_text=text,
            remainder_text="",
            reason="pass_through",
            metadata={
                "source_language": source_language,
                "committed_tail": committed_tail,
            },
        )

    def commit_segment(
        self,
        segment: SpeechSegment,
        *,
        committed_tail: str = "",
    ) -> CommitDecision:
        return self.decide(
            segment.text,
            source_language=segment.source_language,
            committed_tail=committed_tail,
        )
