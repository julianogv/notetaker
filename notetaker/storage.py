"""Disk layout of a meeting and reading/writing of meta.json."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

META_FILENAME = "meta.json"


def slugify(text: str) -> str:
    """Converts a title to a safe ASCII slug for folder name."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return ascii_text or "meeting"


@dataclass
class Meeting:
    """A meeting is the folder on disk that represents it."""

    path: Path

    # --- Paths of tracks and artifacts ---
    @property
    def audio_mic(self) -> Path:
        return self.path / "audio-mic.opus"

    @property
    def audio_system(self) -> Path:
        return self.path / "audio-system.opus"

    @property
    def ffmpeg_log_mic(self) -> Path:
        return self.path / "ffmpeg-mic.log"

    @property
    def ffmpeg_log_system(self) -> Path:
        return self.path / "ffmpeg-system.log"

    @property
    def transcript_mic(self) -> Path:
        return self.path / "transcript-mic.txt"

    @property
    def transcript_system(self) -> Path:
        return self.path / "transcript-system.txt"

    @property
    def transcript_full(self) -> Path:
        return self.path / "transcript-full.txt"

    @property
    def summary_md(self) -> Path:
        return self.path / "summary.md"

    @property
    def meta_path(self) -> Path:
        return self.path / META_FILENAME

    # --- meta.json ---
    def read_meta(self) -> "Meta":
        return Meta.load(self.meta_path)

    def write_meta(self, meta: "Meta") -> None:
        meta.save(self.meta_path)


@dataclass
class Meta:
    """Metadata of a meeting (persisted in meta.json)."""

    title: str = ""
    mode: str = "online"            # online | in-person | listener
    lang: str = "auto"             # input language (transcription)
    output_lang: str = "meeting"   # Summary output language
    diarization: str = "level1"    # level1 | level2
    whisper_model: str = "medium"
    status: str = "created"        # created|recording|transcribing|summarizing|done|error
    created_at: str = ""
    stopped_at: str = ""
    duration_seconds: float = 0.0
    ffmpeg_pids: list[int] = field(default_factory=list)
    mic_source: str = ""
    monitor_source: str = ""
    detected_lang: str = ""        # language detected by Whisper (if auto)
    error: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Meta":
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f: data.get(f) for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def create_meeting(storage_root: Path, title: str) -> Meeting:
    """Creates the meeting folder: <root>/<date_time_slug>/."""
    now = datetime.now()
    folder = f"{now.strftime('%Y-%m-%d_%H%M')}_{slugify(title)}"
    path = storage_root.expanduser() / folder
    path.mkdir(parents=True, exist_ok=True)
    return Meeting(path=path)


def list_meetings(storage_root: Path) -> list[Meeting]:
    """Lists existing meetings, most recent first."""
    root = storage_root.expanduser()
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir() and (p / META_FILENAME).exists()]
    dirs.sort(key=lambda p: p.name, reverse=True)
    return [Meeting(path=p) for p in dirs]


def find_active_meeting(storage_root: Path) -> Meeting | None:
    """Returns the meeting being recorded (status == recording), if any."""
    for meeting in list_meetings(storage_root):
        try:
            if meeting.read_meta().status == "recording":
                return meeting
        except Exception:
            continue
    return None


def resolve_meeting(storage_root: Path, ref: str) -> Meeting | None:
    """Resolves a meeting by absolute path or folder name."""
    p = Path(ref).expanduser()
    if p.is_dir() and (p / META_FILENAME).exists():
        return Meeting(path=p)
    candidate = storage_root.expanduser() / ref
    if candidate.is_dir() and (candidate / META_FILENAME).exists():
        return Meeting(path=candidate)
    return None
