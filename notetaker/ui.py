"""Terminal output helpers: spinner/animation and formatting.

Respects TTY: when output is not a terminal (pipe, file), degrades to simple
single-line messages without cursor escapes.
"""

from __future__ import annotations

import itertools
import shutil
import sys
import threading
import time
from datetime import datetime

# Spinner frames (braille). Fallback to ascii when terminal doesn't support.
_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_FRAMES_ASCII = "|/-\\"


def is_tty() -> bool:
    return sys.stdout.isatty()


def _frames() -> str:
    enc = (sys.stdout.encoding or "").lower()
    if "utf" in enc:
        return _FRAMES
    return _FRAMES_ASCII


def timestamp() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


def log(msg: str) -> None:
    print(f"{timestamp()} {msg}")


def format_duration(seconds: float) -> str:
    """Formats seconds as HH:MM:SS (or MM:SS when < 1h)."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_size(num_bytes: int) -> str:
    """Formats bytes as a readable unit (B, KB, MB, GB)."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def clear_line() -> None:
    if is_tty():
        sys.stdout.write("\r\x1b[2K")
        sys.stdout.flush()


def status_line(text: str) -> None:
    """Writes a rewritable status line (stays on same line in TTY)."""
    width = shutil.get_terminal_size((80, 20)).columns
    text = text[: width - 1]
    if is_tty():
        sys.stdout.write("\r\x1b[2K" + text)
        sys.stdout.flush()
    else:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()


class Spinner:
    """Spinner in thread for indefinite duration tasks (e.g., transcription).

    Usage:
        with Spinner("transcribing..."):
            heavy_work()

    In non-TTY, only prints the message once at startup.
    """

    def __init__(self, message: str, interval: float = 0.1):
        self.message = message
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._final: str | None = None

    def _run(self) -> None:
        for frame in itertools.cycle(_frames()):
            if self._stop.is_set():
                break
            status_line(f"{frame} {self.message}")
            time.sleep(self.interval)

    def start(self) -> "Spinner":
        if is_tty():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        else:
            sys.stdout.write(self.message + "\n")
            sys.stdout.flush()
        return self

    def update(self, message: str) -> None:
        self.message = message

    def stop(self, final_message: str | None = None) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        clear_line()
        if final_message:
            print(final_message)

    def __enter__(self) -> "Spinner":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
