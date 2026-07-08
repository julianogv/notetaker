"""Local batch transcription with faster-whisper, one track at a time.

Language can be fixed (pt/es/en) or 'auto' (Whisper detects). The detected
language of the mic track is used as the effective meeting language.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

# Languages explicitly supported by Notetaker.
SUPPORTED_LANGS = ("pt", "es", "en")


class TranscribeError(RuntimeError):
    pass


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscribeStats:
    """Transcription performance metrics for diagnostics.

    rtf (real-time factor) = transcription_time / audio_duration. rtf < 1 is faster
    than real-time; rtf > 1 is slower (common on CPU with large model).
    """

    audio_seconds: float = 0.0        # duration of transcribed audio
    model_load_seconds: float = 0.0   # time to load the model (0 if cached)
    transcribe_seconds: float = 0.0   # actual transcription time
    segments: int = 0

    @property
    def rtf(self) -> float:
        if self.audio_seconds <= 0:
            return 0.0
        return self.transcribe_seconds / self.audio_seconds


@dataclass
class TrackTranscript:
    text: str
    language: str
    segments: list[Segment] = field(default_factory=list)
    stats: TranscribeStats = field(default_factory=TranscribeStats)


_model_cache: dict[tuple, object] = {}

# Memoized result of compute device detection.
_compute_device: tuple[str, str] | None = None


def detect_compute_device() -> tuple[str, str]:
    """Detects the best (device, compute_type) for faster-whisper.

    Returns ("cuda", "float16") if there is a usable NVIDIA GPU by CTranslate2;
    otherwise ("cpu", "int8"). The result is memoized.

    Note: CTranslate2 (faster-whisper backend) only accelerates on NVIDIA GPUs
    (CUDA). There is no Metal/MPS support, so Macs (Apple Silicon or Intel)
    remain on CPU.
    """
    global _compute_device
    if _compute_device is not None:
        return _compute_device

    device, compute = "cpu", "int8"
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            device, compute = "cuda", "float16"
    except Exception:
        # No ctranslate2/CUDA available: keep CPU.
        pass

    _compute_device = (device, compute)
    return _compute_device


def _load_model(model_name: str, cpu_threads: int = 0, use_cache: bool = True):
    """Loads the faster-whisper model on the best available device.

    Uses NVIDIA GPU (cuda/float16) when detected; otherwise CPU (int8). On CPU,
    cpu_threads=0 lets CTranslate2 decide (all cores); for parallel transcription,
    pass half the cores and use_cache=False, so each track has its own instance
    with its own thread pool. On GPU, cpu_threads is irrelevant and thread
    parallelization does not apply.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscribeError(
            "faster-whisper not installed. Run: pip install -e ."
        ) from exc

    device, compute_type = detect_compute_device()

    def _build():
        if device == "cpu":
            return WhisperModel(
                model_name, device=device, compute_type=compute_type,
                cpu_threads=cpu_threads,
            )
        return WhisperModel(model_name, device=device, compute_type=compute_type)

    if use_cache:
        key = (model_name, device, compute_type, cpu_threads)
        cached = _model_cache.get(key)
        if cached is not None:
            return cached
        model = _build()
        _model_cache[key] = model
        return model

    return _build()


def gpu_available() -> bool:
    """True if transcription will use NVIDIA GPU."""
    return detect_compute_device()[0] == "cuda"


def nvidia_gpu_present() -> bool:
    """Indicates if there is an NVIDIA GPU in the hardware (via nvidia-smi).

    Different from detect_compute_device(), which only returns 'cuda' when CUDA
    libs (ctranslate2 + cublas/cudnn) are already usable at runtime. This check
    detects the hardware before the libs are ready, to guide libcublas installation
    on first run.
    """
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        # nvidia-smi present but not executable (missing driver, etc.): no GPU.
        return False
    return result.returncode == 0 and "GPU" in result.stdout


def load_model(model_name: str, cpu_threads: int = 0):
    """Loads a dedicated model instance (without cache).

    The faster-whisper loader is not thread-safe for concurrent use, so the
    parallel path loads models sequentially with this function and then
    transcribes in threads.
    """
    return _load_model(model_name, cpu_threads=cpu_threads, use_cache=False)


def transcribe_track(
    audio_path: Path,
    model_name: str = "medium",
    language: str = "auto",
    cpu_threads: int = 0,
    use_cache: bool = True,
    preloaded_model=None,
) -> TrackTranscript:
    """Transcribes a track. language 'auto' lets Whisper detect.

    preloaded_model: WhisperModel instance already loaded. Used in the parallel
    path, where models are loaded sequentially first (the faster-whisper loader
    is not thread-safe for concurrent use).
    """
    if not audio_path.exists():
        raise TranscribeError(f"audio not found: {audio_path}")
    if audio_path.stat().st_size == 0:
        raise TranscribeError(
            f"empty audio (0 B): {audio_path.name}. Capture may have failed; "
            f"check the ffmpeg-*.log file in the meeting folder and the devices "
            f"with 'notetaker devices'."
        )

    stats = TranscribeStats()

    if preloaded_model is not None:
        model = preloaded_model
    else:
        t0 = time.monotonic()
        model = _load_model(model_name, cpu_threads=cpu_threads, use_cache=use_cache)
        stats.model_load_seconds = time.monotonic() - t0

    lang_arg = None if language == "auto" else language

    # The call returns a generator; heavy work happens when iterating.
    t1 = time.monotonic()
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=lang_arg,
        vad_filter=True,
    )

    segments: list[Segment] = []
    parts: list[str] = []
    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue
        segments.append(Segment(start=seg.start, end=seg.end, text=text))
        parts.append(text)
    stats.transcribe_seconds = time.monotonic() - t1

    stats.audio_seconds = getattr(info, "duration", 0.0) or 0.0
    stats.segments = len(segments)

    return TrackTranscript(
        text=" ".join(parts).strip(),
        language=info.language,
        segments=segments,
        stats=stats,
    )
