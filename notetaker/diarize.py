"""Diarization: assigns speech segments to who spoke.

Level 1 (no ML): uses separate tracks. Everything from the mic track is "You";
everything from the system track is "Participants". The segments of both tracks
are interleaved by timestamp to produce the transcript-full.

Level 2 (ML, optional): uses whisperx/pyannote on the audio to separate each
speaker individually. Requires the [diarization] extra installed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .transcribe import TrackTranscript

# Labels by language for speakers at level 1.
_LABELS = {
    "pt": {"you": "You", "others": "Participants"},
    "es": {"you": "You", "others": "Participants"},
    "en": {"you": "You", "others": "Participants"},
}


class DiarizeError(RuntimeError):
    pass


@dataclass
class Utterance:
    speaker: str
    start: float
    text: str


def _labels_for(lang: str) -> dict[str, str]:
    return _LABELS.get(lang, _LABELS["pt"])


def build_level1(
    mic: TrackTranscript | None,
    system: TrackTranscript | None,
    lang: str = "pt",
) -> list[Utterance]:
    """Interleaves the mic (You) and system (Participants) segments by time."""
    labels = _labels_for(lang)
    utterances: list[Utterance] = []

    if mic:
        for seg in mic.segments:
            utterances.append(Utterance(labels["you"], seg.start, seg.text))
    if system:
        for seg in system.segments:
            utterances.append(Utterance(labels["others"], seg.start, seg.text))

    utterances.sort(key=lambda u: u.start)
    return utterances


def render_transcript(utterances: list[Utterance]) -> str:
    """Formats utterances as text labeled by speaker."""
    lines = [f"[{u.speaker}] {u.text}" for u in utterances]
    return "\n".join(lines)


def render_plain(transcript: TrackTranscript | None) -> str:
    """Formats a single track as continuous text, without speaker label.

    Used in 'import' mode: the source is a single external file (mobile phone,
    call video), so there is no track separation and therefore no level 1
    diarization. The output is continuous transcript, one segment per line.
    """
    if transcript is None:
        return ""
    return "\n".join(seg.text for seg in transcript.segments)


def build_level2(audio_path, mic_transcript: TrackTranscript, lang: str = "pt"):
    """ML diarization by speaker via whisperx. Optional.

    Returns a list of Utterance with labels by speaker (Speaker 1, 2, ...).
    """
    try:
        import whisperx  # noqa: F401
    except ImportError as exc:
        raise DiarizeError(
            "Level 2 requires the diarization extra. Run: pip install -e '.[diarization]'"
        ) from exc

    raise DiarizeError(
        "Level 2 diarization not yet implemented in this version. Use level1."
    )
