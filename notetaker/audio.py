"""Audio capture via ffmpeg, with backend per platform.

Linux:   PulseAudio/PipeWire backend (`-f pulse`), devices via `pactl`.
macOS:   AVFoundation backend (`-f avfoundation`), devices via ffmpeg. The
         output audio (participants, online mode) requires a virtual device
         like BlackHole, because macOS does not expose the output for capture
         natively.
Windows: DirectShow backend (`-f dshow`), devices via ffmpeg. The output audio
         (participants, online mode) requires a virtual device (VB-CABLE,
         VoiceMeeter) or the "Stereo Mix" from the soundcard, because dshow
         does not expose the output for capture natively.

The public API (resolve_devices, start_recording, stop_recording, read_progress)
is the same on any platform; the correct backend is chosen at runtime.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

from .storage import Meeting


class AudioError(RuntimeError):
    pass


@dataclass
class Devices:
    mic_source: str
    monitor_source: str  # empty in in-person mode

    # ffmpeg input format for this platform ("pulse" or "avfoundation").
    input_format: str = "pulse"


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise AudioError(f"command not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise AudioError(
            f"failed to execute {' '.join(cmd)}: {exc.stderr.strip()}"
        ) from exc
    return out.stdout.strip()


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _is_windows() -> bool:
    return os.name == "nt"


def _detached_popen_kwargs() -> dict:
    """kwargs to detach a subprocess from the current terminal/shell.

    POSIX: new session (setsid), so that terminal Ctrl+C doesn't reach the
    process directly. Windows: new process group, requirement for later
    delivering CTRL_BREAK_EVENT in stop_recording; without a new group, ffmpeg
    would not finalize the opus container correctly.
    """
    if _is_windows():
        # CREATE_NEW_PROCESS_GROUP so existe no subprocess do Windows.
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP")
        return {"creationflags": flags}
    return {"start_new_session": True}


def detached_worker_kwargs() -> dict:
    """kwargs for the background worker (post-stop pipeline), without TTY.

    Unlike recorders, the worker does not need to receive a stop signal; it
    just runs until completion. On Windows, combines new process group with
    DETACHED_PROCESS to fully detach from the shell console. On POSIX,
    new session (setsid).
    """
    if _is_windows():
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP") | getattr(
            subprocess, "DETACHED_PROCESS"
        )
        return {"creationflags": flags}
    return {"start_new_session": True}


def _send_stop_signal(pid: int) -> None:
    """Sends the signal that makes ffmpeg finalize the opus container and exit.

    POSIX: SIGINT to the process. Windows: CTRL_BREAK_EVENT to the process group
    (created with CREATE_NEW_PROCESS_GROUP in start_recording); ffmpeg treats it as
    interruption and writes the opus trailer.
    """
    try:
        if _is_windows():
            # CTRL_BREAK_EVENT so existe no signal do Windows.
            os.kill(pid, getattr(signal, "CTRL_BREAK_EVENT"))
        else:
            os.kill(pid, signal.SIGINT)
    except (ProcessLookupError, OSError):
        pass


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        if _is_macos():
            hint = "brew install ffmpeg"
        elif _is_windows():
            hint = "winget install Gyan.FFmpeg (or choco install ffmpeg)"
        else:
            hint = "sudo apt install ffmpeg"
        raise AudioError(f"ffmpeg not found. Install with: {hint}")


# =========================================================================== #
# Linux Backend (PulseAudio / PipeWire)
# =========================================================================== #
def _linux_check() -> None:
    if shutil.which("pactl") is None:
        raise AudioError("pactl not found. PulseAudio/PipeWire is required.")


def _linux_default_source() -> str:
    return _run(["pactl", "get-default-source"])


def _linux_default_sink() -> str:
    return _run(["pactl", "get-default-sink"])


def _linux_monitor() -> str:
    """The monitor source corresponding to the current default sink."""
    return f"{_linux_default_sink()}.monitor"


def _linux_resolve(mode: str, mic_override: str, monitor_override: str) -> Devices:
    if mode == "listener":
        # Only the system output (participants' voices); mic is not recorded.
        monitor = monitor_override or _linux_monitor()
        return Devices(mic_source="", monitor_source=monitor, input_format="pulse")
    mic = mic_override or _linux_default_source()
    if mode == "in-person":
        return Devices(mic_source=mic, monitor_source="", input_format="pulse")
    monitor = monitor_override or _linux_monitor()
    return Devices(mic_source=mic, monitor_source=monitor, input_format="pulse")


def _linux_describe() -> list[str]:
    lines = [
        "backend:                PulseAudio/PipeWire",
        f"default source (mic):   {_linux_default_source()}",
        f"default sink:           {_linux_default_sink()}",
        f"monitor (system audio): {_linux_monitor()}",
    ]
    return lines


# =========================================================================== #
# macOS Backend (AVFoundation)
# =========================================================================== #
# Name of the virtual device used to capture output audio on Mac.
_MACOS_LOOPBACK_HINTS = ("blackhole", "loopback", "soundflower")

# Line from -list_devices: "[AVFoundation indev @ 0x..] [0] Device Name"
_AVF_DEVICE_RE = re.compile(r"\]\s*\[(\d+)\]\s*(.+)$")


def _macos_list_audio_devices() -> list[tuple[int, str]]:
    """Lists (index, name) of audio devices via avfoundation.

    ffmpeg prints the list to stderr and exits with code != 0 (expected behavior
    of -list_devices), so we don't use check=True here.
    """
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
    )
    text = proc.stderr
    devices: list[tuple[int, str]] = []
    in_audio = False
    for line in text.splitlines():
        low = line.lower()
        if "avfoundation audio devices" in low:
            in_audio = True
            continue
        if "avfoundation video devices" in low:
            in_audio = False
            continue
        if not in_audio:
            continue
        m = _AVF_DEVICE_RE.search(line)
        if m:
            devices.append((int(m.group(1)), m.group(2).strip()))
    return devices


def _macos_find_loopback(devices: list[tuple[int, str]]) -> int | None:
    """Finds the index of a loopback device (BlackHole, etc.)."""
    for idx, name in devices:
        if any(hint in name.lower() for hint in _MACOS_LOOPBACK_HINTS):
            return idx
    return None


def _macos_resolve(mode: str, mic_override: str, monitor_override: str) -> Devices:
    devices = _macos_list_audio_devices()
    if not devices and not mic_override:
        raise AudioError(
            "no audio devices detected via avfoundation. "
            "Check the terminal microphone permissions in "
            "System Settings > Privacy and Security > Microphone."
        )

    # Mic: explicit override, or the first audio device (index 0 is the
    # default input on most Macs).
    mic = mic_override or (str(devices[0][0]) if devices else "0")

    if mode == "in-person":
        return Devices(mic_source=mic, monitor_source="", input_format="avfoundation")

    # Online and listener need output audio, which on Mac comes from a
    # virtual device.
    if monitor_override:
        monitor = monitor_override
    else:
        idx = _macos_find_loopback(devices)
        if idx is None:
            listed = ", ".join(f"[{i}] {n}" for i, n in devices) or "(none)"
            raise AudioError(
                f"mode {mode} on macOS requires a virtual audio device to "
                "capture the participants' voices (macOS does not expose the output "
                "natively). Install BlackHole (https://existential.audio/blackhole/), "
                "create an Aggregate Device with it + your output, and use it as the output "
                "during the meeting.\n"
                f"Audio devices detected: {listed}\n"
                "Or use --mode in-person to record only the microphone."
            )
        monitor = str(idx)

    if mode == "listener":
        # Only the system output (participants' voices); mic is not recorded.
        return Devices(mic_source="", monitor_source=monitor, input_format="avfoundation")

    return Devices(mic_source=mic, monitor_source=monitor, input_format="avfoundation")


def _macos_describe() -> list[str]:
    devices = _macos_list_audio_devices()
    lines = ["backend:                AVFoundation (macOS)"]
    for idx, name in devices:
        lines.append(f"  [{idx}] {name}")
    loop = _macos_find_loopback(devices)
    if loop is not None:
        lines.append(f"loopback (system audio): index {loop}")
    else:
        lines.append(
            "loopback (system audio): not found (install BlackHole for online mode)"
        )
    return lines


# =========================================================================== #
# Windows Backend (DirectShow)
# =========================================================================== #
# Common virtual/loopback devices for capturing output audio.
# "Stereo Mix" is the native loopback of some soundcards (when
# enabled); VB-CABLE and VoiceMeeter are third-party virtual devices.
_WINDOWS_LOOPBACK_HINTS = (
    "stereo mix",
    "mixagem estereo",
    "mixagem estéreo",
    "cable output",
    "cable",
    "voicemeeter",
    "virtual",
    "what u hear",
    "wave out",
)

# Line from dshow -list_devices: '"Device Name" (audio)' or, in
# some ffmpeg versions, '[dshow @ 0x..] "Device Name"'. We extract the name
# between quotes.
_DSHOW_DEVICE_RE = re.compile(r'"([^"]+)"')


def _windows_list_audio_devices() -> list[str]:
    """Lists the names of input audio devices via dshow.

    ffmpeg prints the list to stderr and exits with code != 0 (expected behavior
    of -list_devices), so we don't use check=True here.

    In dshow, devices are referenced by name (not by index), and ffmpeg lists
    video and audio devices together; we filter the audio section.
    """
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "dshow", "-list_devices", "true", "-i", "dummy"],
        capture_output=True,
        text=True,
    )
    text = proc.stderr
    devices: list[str] = []
    in_audio = False
    for line in text.splitlines():
        low = line.lower()
        # Section headers vary between ffmpeg versions:
        # "DirectShow audio devices" / "DirectShow video devices". The video line
        # may contain "(some may be both video and audio devices)", so we check
        # "video devices" first to not misclassify it as audio.
        if "video devices" in low:
            in_audio = False
            continue
        if "audio devices" in low:
            in_audio = True
            continue
        if not in_audio:
            continue
        # Ignore the "Alternative name ..." line that ffmpeg prints.
        if "alternative name" in low:
            continue
        m = _DSHOW_DEVICE_RE.search(line)
        if m:
            devices.append(m.group(1).strip())
    return devices


def _windows_find_loopback(devices: list[str]) -> str | None:
    """Finds the name of a loopback/virtual device (Stereo Mix, etc.)."""
    for name in devices:
        if any(hint in name.lower() for hint in _WINDOWS_LOOPBACK_HINTS):
            return name
    return None


def _windows_resolve(mode: str, mic_override: str, monitor_override: str) -> Devices:
    devices = _windows_list_audio_devices()
    if not devices and not mic_override:
        raise AudioError(
            "no audio devices detected via dshow. Check that the "
            "microphone is connected and enabled in Windows Sound Settings, "
            "and check the terminal app microphone permissions in "
            "Settings > Privacy and Security > Microphone."
        )

    # Mic: explicit override, or the first detected audio device.
    mic = mic_override or (devices[0] if devices else "")
    mic_arg = f"audio={mic}"

    if mode == "in-person":
        return Devices(mic_source=mic_arg, monitor_source="", input_format="dshow")

    # Online and listener need output audio, which on Windows comes from a
    # virtual device or "Stereo Mix" (when the soundcard exposes it and it's
    # enabled).
    if monitor_override:
        monitor = monitor_override
    else:
        found = _windows_find_loopback(devices)
        if found is None:
            listed = ", ".join(f'"{n}"' for n in devices) or "(none)"
            raise AudioError(
                f"mode {mode} on Windows requires a virtual audio device or "
                '"Stereo Mix" to capture the participants\' voices '
                "(dshow does not expose the output natively). Enable Stereo Mix in "
                "Sound Settings > Recording (if your soundcard offers it), or install "
                "VB-CABLE (https://vb-audio.com/Cable/) or VoiceMeeter and route the output "
                "to it.\n"
                f"Audio devices detected: {listed}\n"
                "Or use --mode in-person to record only the microphone."
            )
        monitor = found

    monitor_arg = f"audio={monitor}"
    if mode == "listener":
        # Only the system output (participants' voices); mic is not recorded.
        return Devices(mic_source="", monitor_source=monitor_arg, input_format="dshow")
    return Devices(mic_source=mic_arg, monitor_source=monitor_arg, input_format="dshow")


def _windows_describe() -> list[str]:
    devices = _windows_list_audio_devices()
    lines = ["backend:                DirectShow (Windows)"]
    for name in devices:
        lines.append(f'  "{name}"')
    loop = _windows_find_loopback(devices)
    if loop is not None:
        lines.append(f'loopback (system audio): "{loop}"')
    else:
        lines.append(
            "loopback (system audio): not found (enable Stereo Mix or "
            "install VB-CABLE/VoiceMeeter for online mode)"
        )
    return lines


# =========================================================================== #
# Public API (dispatches to the platform backend)
# =========================================================================== #
def check_dependencies() -> None:
    _check_ffmpeg()
    if _is_macos() or _is_windows():
        return  # avfoundation/dshow come built into ffmpeg
    _linux_check()


def check_device_tooling() -> None:
    """Checks only device detection tools (without ffmpeg)."""
    if _is_macos() or _is_windows():
        _check_ffmpeg()  # on Mac/Windows, the listing uses ffmpeg itself
    else:
        _linux_check()


def resolve_devices(
    mode: str,
    mic_override: str = "",
    monitor_override: str = "",
) -> Devices:
    """Resolves the devices to use at start time.

    Config overrides take priority; empty = automatic detection.
    """
    check_dependencies()
    if _is_macos():
        return _macos_resolve(mode, mic_override, monitor_override)
    if _is_windows():
        return _windows_resolve(mode, mic_override, monitor_override)
    return _linux_resolve(mode, mic_override, monitor_override)


def describe_devices() -> list[str]:
    """Descriptive lines of detected devices, for the `devices` command."""
    check_device_tooling()
    if _is_macos():
        return _macos_describe()
    if _is_windows():
        return _windows_describe()
    return _linux_describe()


def _ffmpeg_track_cmd(source: str, output_path: str, input_format: str) -> list[str]:
    """ffmpeg recording a single source to mono opus."""
    return [
        "ffmpeg",
        "-y",
        "-f", input_format,
        "-i", source,
        "-ac", "1",
        "-c:a", "libopus",
        "-b:a", "24k",
        output_path,
    ]


def import_audio(src, dest) -> None:
    """Extracts/converts audio from an external file to mono opus (24k).

    Accepts both audio files (m4a, mp3, wav, opus, etc.) and video files
    (mp4, mkv, mov, etc.): the `-vn` flag discards any video track and ffmpeg
    extracts only the audio, re-encoding to the same format as recorded tracks
    (mono opus 24k) to keep the pipeline homogeneous.

    Unlike tracks recorded live, here there is a single external source
    (mobile phone, recorder, call video), so there is no speaker separation by
    track — the 'import' mode generates a continuous transcript (without labels).
    """
    _check_ffmpeg()
    from pathlib import Path

    src = Path(src)
    dest = Path(dest)
    if not src.exists():
        raise AudioError(f"file not found: {src}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src),
        "-vn",              # ignore video track; extract audio only
        "-ac", "1",
        "-c:a", "libopus",
        "-b:a", "24k",
        str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        detail = stderr.splitlines()[-1] if stderr else "unknown error"
        raise AudioError(f"failed to extract audio from {src.name}: {detail}")
    if not dest.exists() or dest.stat().st_size == 0:
        raise AudioError(
            f"no audio track found in {src.name}. "
            "Does the file contain audio?"
        )


def start_recording(meeting: Meeting, devices: Devices, mode: str) -> list[int]:
    """Starts recording tracks in background. Returns ffmpeg PIDs.

    One track per ffmpeg process: ensures separate tracks (basis for
    level 1 diarization) and avoids premature mixing. Which tracks are recorded
    depends on the sources resolved for the mode: in-person only has mic, listener
    only has system, online has both.

    Processes run detached from the terminal (POSIX: new session via
    start_new_session; Windows: new process group) so that terminal Ctrl+C doesn't
    reach them directly: shutdown is under exclusive control of stop_recording
    (a single stop signal), ensuring ffmpeg finalizes the opus container correctly.
    stderr goes to a log per track, useful for diagnostics.
    """
    pids: list[int] = []
    fmt = devices.input_format
    popen_kwargs = _detached_popen_kwargs()

    def _spawn(source: str, out_path, log_path) -> int:
        log = open(log_path, "wb")
        proc = subprocess.Popen(
            _ffmpeg_track_cmd(source, str(out_path), fmt),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=log,
            **popen_kwargs,
        )
        return proc.pid

    if devices.mic_source:
        pids.append(
            _spawn(devices.mic_source, meeting.audio_mic, meeting.ffmpeg_log_mic)
        )

    if devices.monitor_source:
        pids.append(
            _spawn(
                devices.monitor_source,
                meeting.audio_system,
                meeting.ffmpeg_log_system,
            )
        )

    return pids


def stop_recording(pids: list[int], timeout: float = 10.0) -> None:
    """Stops ffmpeg processes with the stop signal and waits for each to finish.

    The signal (POSIX: SIGINT; Windows: CTRL_BREAK_EVENT) makes ffmpeg write the
    opus container trailer and exit. We send only one signal per process and
    wait for it to finish, to avoid corrupting the output with a second signal.
    """
    for pid in pids:
        _send_stop_signal(pid)

    # Waits for each process to exit (until timeout), to ensure the opus
    # container was finalized before transcribing.
    deadline = time.time() + timeout
    for pid in pids:
        while is_running(pid) and time.time() < deadline:
            time.sleep(0.1)


def _windows_is_running(pid: int) -> bool:
    """Checks if a PID is active on Windows via OpenProcess/GetExitCodeProcess."""
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259

    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def is_running(pid: int) -> bool:
    if _is_windows():
        return _windows_is_running(pid)
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


# Extracts the last "size=  123kB" and the last "time=00:00:12.34" from ffmpeg log.
_SIZE_RE = re.compile(r"size=\s*(\d+)\s*([kKmMgG]?i?)B")
_TIME_RE = re.compile(r"time=\s*(\d+):(\d+):(\d+(?:\.\d+)?)")

_SIZE_UNIT = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}


def read_progress(log_paths: list) -> tuple[int, float]:
    """Reads the size (bytes) and time (seconds) captured from ffmpeg logs.

    The opus muxer only flushes data to the file when finalizing, so disk size
    remains 0 during recording. The ffmpeg log, however, reports the current
    'size=' and 'time=' — we use it as the live source of truth.

    Returns (total_bytes, max_time) by aggregating all tracks.
    """
    total_bytes = 0
    max_time = 0.0
    for path in log_paths:
        try:
            data = open(path, "rb").read()
        except OSError:
            continue
        text = data.decode("utf-8", "ignore")

        size_matches = _SIZE_RE.findall(text)
        if size_matches:
            num, unit = size_matches[-1]
            total_bytes += int(num) * _SIZE_UNIT.get(unit.lower().rstrip("i"), 1)

        time_matches = _TIME_RE.findall(text)
        if time_matches:
            h, m, s = time_matches[-1]
            secs = int(h) * 3600 + int(m) * 60 + float(s)
            max_time = max(max_time, secs)

    return total_bytes, max_time
