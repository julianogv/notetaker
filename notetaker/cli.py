"""Notetaker CLI: start, stop, status, list, devices, summarize."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from . import audio
from . import transcribe
from . import ui
from .config import (
    CONFIG_PATH,
    Config,
    config_exists,
    load_config,
    resolve_llm_command,
    write_config,
)
from .prompts import resolve_output_language
from .storage import (
    Meeting,
    create_meeting,
    find_active_meeting,
    list_meetings,
    resolve_meeting,
)
from .storage import Meta


def _err(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 1


def _audio_size(meeting: Meeting) -> int:
    """Sum the byte size of existing audio tracks."""
    total = 0
    for p in (meeting.audio_mic, meeting.audio_system):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def _run_processing(meeting: Meeting) -> int:
    """Run the pipeline in foreground with spinner per phase."""
    from . import pipeline

    spinner = ui.Spinner("processing...").start()

    def on_progress(phase: str, message: str) -> None:
        if phase == "done":
            return
        # Phases 'stats_*' and 'device' are diagnostics: print permanent line.
        if phase.startswith("stats_") or phase == "device":
            spinner.stop()
            print(message)
            spinner.start()
            return
        spinner.update(message)

    try:
        pipeline.process_meeting(meeting, progress=on_progress)
    except Exception as exc:  # noqa: BLE001
        spinner.stop(f"{ui.timestamp()} processing failed: {exc}")
        return 1
    spinner.stop(f"{ui.timestamp()} summary ready: {meeting.summary_md}")
    return 0


# Recording modes and the live mode-switch menu keys.
_MODE_KEYS = {"o": "online", "i": "in-person", "l": "listener"}


class _KeyReader:
    """Reads single keystrokes (without Enter) during the watch, restoring the
    terminal on exit.

    No-op when stdin is not a TTY (pipe, chained script): in that case the watch
    keeps working with Ctrl+C only, and getch just sleeps to preserve the loop's
    pacing.

    POSIX: cbreak mode via termios — turns off echo and canonical mode, but keeps
    ISIG, so terminal Ctrl+C still raises KeyboardInterrupt (the normal way to
    stop the recording). Windows: msvcrt.
    """

    def __init__(self) -> None:
        self._is_windows = audio._is_windows()
        self.enabled = sys.stdin.isatty()
        self._posix = self.enabled and not self._is_windows
        self._fd: int | None = None
        self._old = None

    def __enter__(self) -> "_KeyReader":
        if self._posix:
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc) -> None:
        if self._posix and self._old is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def getch(self, timeout: float = 0.0) -> str | None:
        """Returns a key if pressed within `timeout`s; otherwise None.

        Outside a TTY, sleeps `timeout` and returns None (keeps the loop's pace).
        """
        if not self.enabled:
            time.sleep(timeout)
            return None
        if self._posix:
            import select

            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if ready:
                return sys.stdin.read(1)
            return None
        # Windows: poll msvcrt until a key or the timeout.
        import msvcrt

        deadline = time.monotonic() + timeout
        while True:
            if msvcrt.kbhit():
                return msvcrt.getwch()
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.01)


def _apply_mode_switch(meeting: Meeting, new_mode: str, cfg: Config) -> str:
    """Turns tracks on/off to reflect `new_mode` without stopping the recording.

    A mode only defines which tracks are recorded (mic and/or system). Switching
    mode therefore means turning on the missing tracks and turning off the extra
    ones. A track turned on now gets silence padding (pad_seconds) equal to the
    elapsed time, to stay aligned with t=0 of the others (diarization assumes
    tracks start together).

    Limitation: turning a track back on that already recorded before (e.g.
    online->listener->online) overwrites its file (ffmpeg -y), losing the earlier
    part. The common case (a single switch during the meeting) is unaffected.

    Returns a result message to show the user.
    """
    meta = meeting.read_meta()
    if meta.mode == new_mode:
        return f"already in {new_mode} mode."
    try:
        devices = audio.resolve_devices(
            new_mode, cfg.audio.mic_source, cfg.audio.monitor_source
        )
    except audio.AudioError as exc:
        return f"failed to switch mode: {exc}"

    try:
        started = datetime.fromisoformat(meta.created_at)
        elapsed = max(0.0, (datetime.now() - started).total_seconds())
    except Exception:
        elapsed = 0.0

    fmt = devices.input_format

    # mic track
    want_mic = bool(devices.mic_source)
    have_mic = bool(meta.mic_pid) and audio.is_running(meta.mic_pid)
    if want_mic and not have_mic:
        meta.mic_pid = audio.start_track(
            devices.mic_source, meeting.audio_mic, meeting.ffmpeg_log_mic, fmt,
            pad_seconds=elapsed,
        )
        meta.mic_source = devices.mic_source
    elif have_mic and not want_mic:
        audio.stop_track(meta.mic_pid)
        meta.mic_pid = 0

    # system track
    want_sys = bool(devices.monitor_source)
    have_sys = bool(meta.system_pid) and audio.is_running(meta.system_pid)
    if want_sys and not have_sys:
        meta.system_pid = audio.start_track(
            devices.monitor_source, meeting.audio_system, meeting.ffmpeg_log_system,
            fmt, pad_seconds=elapsed,
        )
        meta.monitor_source = devices.monitor_source
    elif have_sys and not want_sys:
        audio.stop_track(meta.system_pid)
        meta.system_pid = 0

    meta.mode = new_mode
    meta.ffmpeg_pids = [p for p in (meta.mic_pid, meta.system_pid) if p]
    meeting.write_meta(meta)
    return f"mode switched to {new_mode}."


def _mode_switch_menu(meeting: Meeting, cfg: Config, reader: _KeyReader) -> None:
    """Mode-switch menu triggered by the 'm' key during the watch."""
    meta = meeting.read_meta()
    ui.clear_line()
    print(f"\nswitch mode (current: {meta.mode}):")
    print("  [o] online   [i] in-person   [l] listener   [esc] cancel")
    while True:
        ch = reader.getch(timeout=15.0)
        if ch is None or ch in ("\x1b", "\r", "\n", "c"):  # esc/enter/c/timeout
            print("cancelled.")
            return
        new_mode = _MODE_KEYS.get(ch.lower())
        if new_mode is None:
            continue  # invalid key: wait for another
        print(_apply_mode_switch(meeting, new_mode, cfg))
        return


def _watch_recording(meeting: Meeting, cfg: Config) -> None:
    """Live monitoring: spinner + elapsed time + audio size + mode.

    Blocks until Ctrl+C. Does not stop the recording here (caller handles it).
    In a TTY, the 'm' key opens the mode-switch menu without stopping recording.

    Size/time come from ffmpeg logs (read_progress), which report current progress
    continuously (the file on disk also grows live thanks to -flush_packets).
    """
    logs = [meeting.ffmpeg_log_mic, meeting.ffmpeg_log_system]
    frames = ui._frames()
    i = 0
    current_mode = meeting.read_meta().mode
    with _KeyReader() as reader:
        hint = ("m switch mode, Ctrl+C to stop"
                if reader.enabled else "Ctrl+C to stop")
        while True:
            frame = frames[i % len(frames)]
            size, elapsed = audio.read_progress(logs)
            ui.status_line(
                f"{frame} recording [{current_mode}]  {ui.format_duration(elapsed)}  "
                f"audio: {ui.format_size(size)}  ({hint})"
            )
            i += 1
            ch = reader.getch(timeout=0.2)
            if ch and ch.lower() == "m":
                _mode_switch_menu(meeting, cfg, reader)
                current_mode = meeting.read_meta().mode


def _dispatch_background_processing(meeting: Meeting) -> None:
    """Dispatch the processing pipeline to background, detached from shell."""
    subprocess.Popen(
        [sys.executable, "-m", "notetaker.cli", "_process", str(meeting.path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **audio.detached_worker_kwargs(),
    )


def _finalize_stopped_meta(meeting: Meeting) -> Meta:
    """Stop ffmpeg and mark meta as 'transcribing'. Returns the meta."""
    meta = meeting.read_meta()
    audio.stop_recording(meta.ffmpeg_pids)  # wait for ffmpeg to finalize opus
    meta.stopped_at = datetime.now().isoformat(timespec="seconds")
    try:
        started = datetime.fromisoformat(meta.created_at)
        meta.duration_seconds = (datetime.now() - started).total_seconds()
    except Exception:
        pass
    meta.status = "transcribing"
    meta.ffmpeg_pids = []
    meeting.write_meta(meta)
    return meta


def _prompt_active_conflict(active: Meeting, live: bool = True) -> str:
    """Ask what to do when a meeting is already recording (or was interrupted).

    `live` indicates whether the recording is still in progress (ffmpeg alive) or
    it crashed. Returns 'stop' (stop current and start new), 'stop_only' (stop
    current only, don't start new), 'new' (start in new folder, keep current
    recording), or 'abort' (cancel). Only called when there's an interactive
    terminal.
    """
    if live:
        print(f"a meeting is already recording: {active.path.name}.")
    else:
        print(
            f"there is an interrupted recording (it crashed): {active.path.name}. "
            "The audio on disk was preserved."
        )
    print("  [s] stop current recording and start a new one")
    print("  [o] stop current recording only (don't start new)")
    print("  [n] start in a new folder (keep current recording)")
    print("  [c] cancel")
    while True:
        try:
            resp = input("what would you like to do? [s/o/n/c]: ").strip().lower()
        except EOFError:
            return "abort"
        if resp in ("s", "stop"):
            return "stop"
        if resp in ("o", "only"):
            return "stop_only"
        if resp in ("n", "new"):
            return "new"
        if resp in ("c", "cancel", ""):
            return "abort"
        print("  invalid option. Choose s, o, n, or c.")


# --------------------------------------------------------------------------- #
# start
# --------------------------------------------------------------------------- #
def cmd_start(args: argparse.Namespace) -> int:
    cfg = load_config()

    active = find_active_meeting(cfg.storage_root)
    if active:
        live = audio.any_recording_alive(active.read_meta().ffmpeg_pids)
        # Without interactive terminal (script, pipe, --no-watch chained) cannot
        # ask: maintain safe behavior of not touching the running recording.
        if not sys.stdin.isatty():
            if live:
                return _err(
                    f"a meeting is already recording: {active.path.name}. "
                    "Run 'notetaker stop'."
                )
            return _err(
                f"there is an interrupted recording: {active.path.name}. "
                "Run 'notetaker list --retry' to recover it (or 'notetaker stop')."
            )
        decision = _prompt_active_conflict(active, live)
        if decision == "abort":
            print("cancelled; the running recording was kept.")
            return 1
        if decision == "stop_only":
            ui.log(f"stopping current meeting: {active.path.name}...")
            _finalize_stopped_meta(active)
            _dispatch_background_processing(active)
            ui.log("meeting stopped; processing in background. "
                   "Check with 'notetaker status'.")
            return 0
        if decision == "stop":
            ui.log(f"stopping current meeting: {active.path.name}...")
            _finalize_stopped_meta(active)
            _dispatch_background_processing(active)
            ui.log("previous meeting stopped; processing in background.")
        # decision == "new": proceed and create a new folder (current keeps
        # recording; next 'stop' ends the most recent one).

    try:
        devices = audio.resolve_devices(
            args.mode, cfg.audio.mic_source, cfg.audio.monitor_source
        )
    except audio.AudioError as exc:
        return _err(str(exc))

    meeting = create_meeting(cfg.storage_root, args.title)
    meta = Meta(
        title=args.title,
        mode=args.mode,
        lang=args.lang or cfg.whisper.language,
        output_lang=args.output_lang or cfg.summary.language,
        diarization=args.diarization,
        whisper_model=cfg.whisper.model,
        status="recording",
        created_at=datetime.now().isoformat(timespec="seconds"),
        mic_source=devices.mic_source,
        monitor_source=devices.monitor_source,
        extra={"llm_command": resolve_llm_command(cfg.llm.provider, cfg.llm.model)},
    )

    try:
        pids = audio.start_recording(meeting, devices, args.mode)
    except audio.AudioError as exc:
        meta.status = "error"
        meta.error = str(exc)
        meeting.write_meta(meta)
        return _err(str(exc))

    meta.mic_pid = pids["mic"]
    meta.system_pid = pids["system"]
    meta.ffmpeg_pids = [p for p in (pids["mic"], pids["system"]) if p]
    meeting.write_meta(meta)

    ui.log(f"recording: {meeting.path.name}")
    print(f"  mode: {args.mode}")
    if devices.mic_source:
        print(f"  mic: {devices.mic_source}")
    if devices.monitor_source:
        print(f"  system: {devices.monitor_source}")
    print("  (Ctrl+C stops recording and generates summary)")

    # Detached mode: return and let user stop with 'notetaker stop'.
    if not args.watch:
        ui.log("run 'notetaker stop' to stop recording and generate the summary.")
        return 0

    if sys.stdin.isatty():
        print("  (press 'm' to switch recording mode without stopping)")

    # Watch mode (default): live monitoring until Ctrl+C, then process.
    try:
        _watch_recording(meeting, cfg)
    except KeyboardInterrupt:
        pass

    ui.clear_line()
    print(f"\n{ui.timestamp()} stopping recording...")
    return _finish_recording(meeting)


def _finish_recording(meeting: Meeting) -> int:
    """Stop ffmpeg, update metadata, and start foreground processing."""
    meta = meeting.read_meta()

    # Stop ffmpeg with feedback (opus container finalization can take a few
    # seconds on long recordings).
    spinner = ui.Spinner("finalizing audio files...").start()
    audio.stop_recording(meta.ffmpeg_pids)  # wait for ffmpeg to finalize opus
    spinner.stop()

    meta.stopped_at = datetime.now().isoformat(timespec="seconds")
    try:
        started = datetime.fromisoformat(meta.created_at)
        meta.duration_seconds = (datetime.now() - started).total_seconds()
    except Exception:
        pass
    meta.status = "transcribing"
    meta.ffmpeg_pids = []
    meeting.write_meta(meta)

    ui.log(f"recording stopped: {meeting.path.name} "
           f"({ui.format_duration(meta.duration_seconds)}, "
           f"{ui.format_size(_audio_size(meeting))})")
    ui.log("starting local transcription and summary generation...")
    return _run_processing(meeting)


# --------------------------------------------------------------------------- #
# stop
# --------------------------------------------------------------------------- #
def cmd_stop(args: argparse.Namespace) -> int:
    cfg = load_config()
    meeting = find_active_meeting(cfg.storage_root)
    if meeting is None:
        return _err("no meeting is currently recording.")

    _finalize_stopped_meta(meeting)

    ui.log(f"recording stopped: {meeting.path.name}")

    if args.wait:
        return _run_processing(meeting)

    _dispatch_background_processing(meeting)
    ui.log("processing in background. Check with 'notetaker status'.")
    return 0


# --------------------------------------------------------------------------- #
# _process (internal, called in background by stop)
# --------------------------------------------------------------------------- #
def cmd_process(args: argparse.Namespace) -> int:
    from . import pipeline

    meeting = Meeting(path=__import__("pathlib").Path(args.path))
    try:
        pipeline.process_meeting(meeting)
    except Exception:  # noqa: BLE001
        return 1
    return 0


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def cmd_status(args: argparse.Namespace) -> int:
    cfg = load_config()
    meetings = list_meetings(cfg.storage_root)
    if not meetings:
        print("no meetings found.")
        return 0

    latest = meetings[0]
    meta = latest.read_meta()
    print(f"meeting: {latest.path.name}")
    print(f"  status: {meta.status}")
    print(f"  mode: {meta.mode} | diarization: {meta.diarization}")
    if meta.detected_lang:
        print(f"  detected language: {meta.detected_lang}")
    if meta.error:
        print(f"  error: {meta.error}")
    if meta.status == "done":
        print(f"  summary: {latest.summary_md}")
    return 0


# --------------------------------------------------------------------------- #
# list / pending summaries / crash recovery
# --------------------------------------------------------------------------- #
def _nonempty(path: Path) -> bool:
    """True if the file exists and has content (> 0 bytes)."""
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def _summary_category(meeting: Meeting, meta: Meta) -> str:
    """Categorizes a meeting by processing state, to guide the action:

    'recording'             -> recording in progress (ffmpeg alive; don't touch).
    'interrupted recording' -> meta status 'recording' but no ffmpeg alive: the
                               recording crashed (kill/hibernation/power loss). The
                               audio on disk survived (-flush_packets). Acted on by
                               --retry.
    'ok'                    -> summary.md already exists and is not empty.
    'summary pending'       -> transcript exists, but the summary wasn't finished
                               (e.g. the laptop died during summarization).
                               Acted on by --summarize.
    'transcription pending' -> audio exists, but the transcript is empty/missing
                               (nothing to summarize yet). Acted on by --retry.
    'no audio'              -> neither audio nor transcript.
    """
    if meta.status == "recording":
        if audio.any_recording_alive(meta.ffmpeg_pids):
            return "recording"
        return "interrupted recording"
    if _nonempty(meeting.summary_md):
        return "ok"
    if _nonempty(meeting.transcript_full):
        return "summary pending"
    if meeting.audio_mic.exists() or meeting.audio_system.exists():
        return "transcription pending"
    return "no audio"


def _meetings_in(path: Path) -> list[Meeting]:
    """Meetings inside `path`.

    Accepts both a storage root (several meetings in subfolders) and the path of
    a single meeting (the folder itself has meta.json).
    """
    path = path.expanduser()
    if Meeting(path=path).meta_path.exists():
        return [Meeting(path=path)]
    return list_meetings(path)


def _generate_summary_for(
    meeting: Meeting, cfg: Config, output_lang: str = ""
) -> Path:
    """Generate/regenerate summary.md from the existing transcript-full.

    Returns the summary path. Raises RuntimeError if there is no usable
    transcript. Marks the meeting as 'done' and clears any prior error on success.
    """
    from .summarize import generate_summary

    if not _nonempty(meeting.transcript_full):
        raise RuntimeError("transcript-full.txt missing or empty")

    meta = meeting.read_meta()
    transcript = meeting.transcript_full.read_text(encoding="utf-8")
    out_lang = resolve_output_language(
        output_lang or meta.output_lang, meta.detected_lang or meta.lang
    )
    llm_command = meta.extra.get(
        "llm_command", resolve_llm_command(cfg.llm.provider, cfg.llm.model)
    )
    md = generate_summary(transcript, llm_command, out_lang, title=meta.title)

    meeting.summary_md.write_text(md, encoding="utf-8")
    meta.status = "done"
    meta.error = ""
    meeting.write_meta(meta)
    return meeting.summary_md


def _summarize_pending(
    pend: list[Meeting], cfg: Config, output_lang: str, root: Path
) -> int:
    """Generate, in sequence, the summaries of meetings with a pending summary."""
    if not pend:
        print(f"no pending summaries in {root}.")
        return 0
    ui.log(f"generating {len(pend)} pending summary(ies)...")
    ok = 0
    for m in pend:
        spinner = ui.Spinner(f"summarizing {m.path.name}...").start()
        try:
            _generate_summary_for(m, cfg, output_lang)
            spinner.stop(f"  ok: {m.path.name}")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            spinner.stop(f"  failed: {m.path.name} ({exc})")
    ui.log(f"{ok}/{len(pend)} summary(ies) generated.")
    return 0


def _finalize_crashed_meta(meeting: Meeting, meta: Meta) -> None:
    """Safely finalize a recording interrupted by a crash.

    Signals any leftover ffmpeg (stop_recording ignores stale PIDs), derives
    stopped_at/duration from the log (the real captured time), and clears the
    PIDs. Leaves the meta ready to reprocess from the audio left on disk.
    """
    audio.stop_recording(meta.ffmpeg_pids)  # only hits live ffmpeg; rest is no-op
    _bytes, elapsed = audio.read_progress(
        [meeting.ffmpeg_log_mic, meeting.ffmpeg_log_system]
    )
    if elapsed and not meta.duration_seconds:
        meta.duration_seconds = elapsed
    if not meta.stopped_at:
        meta.stopped_at = datetime.now().isoformat(timespec="seconds")
    meta.ffmpeg_pids = []
    meta.mic_pid = 0
    meta.system_pid = 0


def _retry_pending(pend: list[Meeting], cfg: Config) -> int:
    """Reprocess, in sequence (one at a time), the meetings that have audio but
    didn't reach a summary: transcription pending and interrupted recordings.

    Sequential on purpose: each retry runs Whisper, so processing in parallel
    would saturate CPU/GPU. Runs in the foreground, showing each meeting's
    progress, instead of dispatching to background.
    """
    if not pend:
        print("nothing to reprocess.")
        return 0
    ui.log(f"reprocessing {len(pend)} recording(s) (one at a time)...")
    ok = 0
    for i, m in enumerate(pend, 1):
        print(f"\n[{i}/{len(pend)}] {m.path.name}")
        meta = m.read_meta()
        if meta.status == "recording":
            # interrupted recording: safely finalize before processing.
            _finalize_crashed_meta(m, meta)
        meta.error = ""
        meta.status = "transcribing"
        m.write_meta(meta)
        if _run_processing(m) == 0:
            ok += 1
    ui.log(f"{ok}/{len(pend)} reprocessed successfully.")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    cfg = load_config()
    root = Path(args.dir).expanduser() if args.dir else cfg.storage_root
    meetings = _meetings_in(root)
    if not meetings:
        print(f"no meetings found in {root}")
        return 0

    rows = []
    for m in meetings:
        meta = m.read_meta()
        rows.append((m, meta, _summary_category(m, meta)))
    pend_summary = [m for m, _meta, cat in rows if cat == "summary pending"]
    pend_transc = [m for m, _meta, cat in rows if cat == "transcription pending"]
    crashed = [m for m, _meta, cat in rows if cat == "interrupted recording"]
    # --retry recovers everything that has audio but didn't reach a summary.
    recoverable = crashed + pend_transc

    if args.retry:
        return _retry_pending(recoverable, cfg)
    if args.summarize:
        return _summarize_pending(pend_summary, cfg, args.output_lang, root)

    shown = rows
    if args.pending:
        shown = [r for r in rows if r[2] == "summary pending"]
    elif args.pending_transcription:
        shown = [r for r in rows if r[2] == "transcription pending"]
    if not shown:
        print(f"nothing to show in {root}.")
        return 0

    for m, meta, cat in shown:
        print(f"{m.path.name:48s} [{meta.status:12s}] {cat}")

    # Action hints (only in the full, unfiltered listing).
    if not args.pending and not args.pending_transcription:
        if pend_summary:
            print(
                f"\n{len(pend_summary)} with a pending summary. "
                "Run 'notetaker list --summarize' to generate them."
            )
        if recoverable:
            detail = []
            if pend_transc:
                detail.append(f"{len(pend_transc)} transcription pending")
            if crashed:
                detail.append(f"{len(crashed)} interrupted recording")
            print(
                f"{len(recoverable)} to reprocess ({', '.join(detail)}). "
                "Run 'notetaker list --retry'."
            )
    return 0


# --------------------------------------------------------------------------- #
# devices
# --------------------------------------------------------------------------- #
def cmd_devices(args: argparse.Namespace) -> int:
    try:
        for line in audio.describe_devices():
            print(line)
    except audio.AudioError as exc:
        return _err(str(exc))
    return 0


# --------------------------------------------------------------------------- #
# summarize (regenerate from existing transcript)
# --------------------------------------------------------------------------- #
def cmd_summarize(args: argparse.Namespace) -> int:
    cfg = load_config()
    meeting = resolve_meeting(cfg.storage_root, args.folder)
    if meeting is None:
        return _err(f"meeting not found: {args.folder}")

    try:
        path = _generate_summary_for(meeting, cfg, args.output_lang)
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))

    ui.log(f"summary regenerated: {path}")
    return 0


# --------------------------------------------------------------------------- #
# retry (reprocess from transcription after a failure)
# --------------------------------------------------------------------------- #
def cmd_retry(args: argparse.Namespace) -> int:
    cfg = load_config()
    meeting = resolve_meeting(cfg.storage_root, args.folder)
    if meeting is None:
        return _err(f"meeting not found: {args.folder}")
    if not meeting.audio_mic.exists() and not meeting.audio_system.exists():
        return _err("no audio tracks found; nothing to reprocess.")

    meta = meeting.read_meta()
    meta.error = ""
    meta.status = "transcribing"
    meeting.write_meta(meta)

    ui.log(f"reprocessing: {meeting.path.name}")

    if args.wait:
        return _run_processing(meeting)

    _dispatch_background_processing(meeting)
    ui.log("processing in background. Check with 'notetaker status'.")
    return 0


# --------------------------------------------------------------------------- #
# import (transcribe and summarize a single external audio/video file)
# --------------------------------------------------------------------------- #
def cmd_import(args: argparse.Namespace) -> int:
    """Import an external file (audio or video), transcribe, and generate summary.

    The source can be recorded elsewhere (phone, recorder, video call). The audio
    is extracted/converted to the track format (opus mono) and the meeting is
    processed by the same batch pipeline. Since there is only one source, there
    is no speaker separation per track: 'import' mode generates continuous
    transcription (without labels).
    """
    cfg = load_config()

    src = Path(args.file).expanduser()
    if not src.exists():
        return _err(f"file not found: {src}")
    if not src.is_file():
        return _err(f"not a file: {src}")

    title = args.title or src.stem
    meeting = create_meeting(cfg.storage_root, title)
    meta = Meta(
        title=title,
        mode="import",
        lang=args.lang or cfg.whisper.language,
        output_lang=args.output_lang or cfg.summary.language,
        diarization="level1",
        whisper_model=cfg.whisper.model,
        status="transcribing",
        created_at=datetime.now().isoformat(timespec="seconds"),
        extra={
            "llm_command": resolve_llm_command(cfg.llm.provider),
            "source_file": str(src),
        },
    )
    meeting.write_meta(meta)

    # Extract audio to mic track (single source). Discard video if present.
    spinner = ui.Spinner(f"extracting audio from {src.name}...").start()
    try:
        audio.import_audio(src, meeting.audio_mic)
    except audio.AudioError as exc:
        spinner.stop()
        meta.status = "error"
        meta.error = str(exc)
        meeting.write_meta(meta)
        return _err(str(exc))
    spinner.stop(f"{ui.timestamp()} audio imported: {meeting.path.name}")

    ui.log("starting local transcription and summary generation...")

    if args.wait:
        return _run_processing(meeting)

    _dispatch_background_processing(meeting)
    ui.log("processing in background. Check with 'notetaker status'.")
    return 0


# --------------------------------------------------------------------------- #
# setup (interactive configuration assistant)
# --------------------------------------------------------------------------- #
def _prompt(label: str, default: str, choices: list[str] | None = None) -> str:
    """Read a user response with default value.

    Enter accepts the default. When 'choices' are present, repeats until the
    response is valid (case-insensitive). Empty is a valid response (keeps default).
    """
    hint = f" [{'/'.join(choices)}]" if choices else ""
    suffix = f" (default: {default})" if default else " (default: empty = auto)"
    while True:
        try:
            response = input(f"{label}{hint}{suffix}: ").strip()
        except EOFError:
            return default
        if not response:
            return default
        if choices and response.lower() not in [c.lower() for c in choices]:
            print(f"  invalid option. Choose one of: {', '.join(choices)}")
            continue
        return response


def _check_gpu_setup() -> None:
    """On first run, warn about CUDA library when NVIDIA GPU is present.

    CTranslate2 (faster-whisper backend) only accelerates on NVIDIA GPUs and
    depends on libcublas. If hardware has GPU but libraries aren't yet usable,
    guide installation to enable acceleration.
    """
    if not transcribe.nvidia_gpu_present():
        return
    if transcribe.gpu_available():
        # GPU already usable (libs present): nothing to do.
        print("NVIDIA GPU detected and ready to accelerate transcription.\n")
        return

    print("\nNVIDIA GPU detected, but CUDA library (libcublas) is not ready.")
    print("Install it to accelerate transcription:")
    print("  sudo apt-get install -y libcublas-12-0\n")


def cmd_setup(args: argparse.Namespace) -> int:
    """Interactive assistant: asks each option with default value and saves config."""
    # Start from existing config (if any) to preserve current values.
    base = load_config() if config_exists() else Config()

    print("Notetaker configuration")
    print(f"config will be saved to: {CONFIG_PATH}")
    print("press Enter to accept the default value in parentheses.\n")

    storage = _prompt("meeting folder (storage_root)", str(base.storage_root))

    print("\n-- audio (leave empty for auto-detection on 'start') --")
    mic_source = _prompt("microphone device (mic_source)", base.audio.mic_source)
    monitor_source = _prompt(
        "system audio device (monitor_source)", base.audio.monitor_source
    )

    print("\n-- transcription (whisper) --")
    model = _prompt(
        "Whisper model", base.whisper.model,
        choices=["tiny", "base", "small", "medium", "large-v3"],
    )
    language = _prompt(
        "language spoken in meetings", base.whisper.language,
        choices=["auto", "pt", "es", "en"],
    )

    print("\n-- summary --")
    summary_language = _prompt(
        "summary language", base.summary.language,
        choices=["meeting", "pt", "es", "en"],
    )

    print("\n-- LLM (CLI that receives transcript via stdin) --")
    llm_provider = _prompt(
        "LLM Provider", base.llm.provider,
        choices=["kiro", "claude"],
    ).lower()
    llm_model = _prompt(
        "LLM model (empty = CLI default)", base.llm.model,
    )

    cfg = Config(
        storage_root=Path(storage).expanduser(),
        audio=type(base.audio)(mic_source=mic_source, monitor_source=monitor_source),
        whisper=type(base.whisper)(model=model, language=language),
        summary=type(base.summary)(language=summary_language),
        llm=type(base.llm)(provider=llm_provider, model=llm_model),
    )
    path = write_config(cfg)
    print(f"\nconfig saved to {path}")
    print("done. Use: notetaker start \"my meeting\"")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="notetaker", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("start", help="start recording a meeting")
    s.add_argument("title", help="meeting title")
    s.add_argument(
        "--mode", choices=["online", "in-person", "listener"], default="online"
    )
    s.add_argument("--lang", choices=["auto", "pt", "es", "en"], default="")
    s.add_argument("--diarization", choices=["level1", "level2"], default="level1")
    s.add_argument("--output-lang", dest="output_lang",
                   choices=["meeting", "pt", "es", "en"], default="")
    s.add_argument("--no-watch", dest="watch", action="store_false",
                   help="do not monitor live; return and wait for 'notetaker stop'")
    s.set_defaults(func=cmd_start, watch=True)

    st = sub.add_parser("stop", help="stop recording and generate summary")
    st.add_argument("--wait", action="store_true",
                    help="process in foreground instead of background")
    st.set_defaults(func=cmd_stop)

    stt = sub.add_parser("status", help="show status of the most recent meeting")
    stt.set_defaults(func=cmd_status)

    ls = sub.add_parser(
        "list", help="list meetings (and optionally generate pending summaries)"
    )
    ls.add_argument("--dir", default="",
                    help="folder to list (default: storage_root from config); "
                         "accepts the root or the path of a single meeting")
    ls.add_argument("--pending", action="store_true",
                    help="show only meetings with a pending summary")
    ls.add_argument("--pending-transcription", dest="pending_transcription",
                    action="store_true",
                    help="show only meetings with a pending transcription")
    ls_action = ls.add_mutually_exclusive_group()
    ls_action.add_argument("--summarize", action="store_true",
                           help="generate the pending summaries in the folder")
    ls_action.add_argument("--retry", action="store_true",
                           help="reprocess (one at a time) meetings with pending "
                                "transcription or interrupted recordings")
    ls.add_argument("--output-lang", dest="output_lang",
                    choices=["meeting", "pt", "es", "en"], default="")
    ls.set_defaults(func=cmd_list)

    dv = sub.add_parser("devices", help="show detected audio devices")
    dv.set_defaults(func=cmd_devices)

    su = sub.add_parser("setup", help="interactive configuration assistant")
    su.set_defaults(func=cmd_setup)

    sm = sub.add_parser("summarize", help="regenerate summary from transcript")
    sm.add_argument("folder", help="meeting folder (name or path)")
    sm.add_argument("--output-lang", dest="output_lang",
                    choices=["meeting", "pt", "es", "en"], default="")
    sm.set_defaults(func=cmd_summarize)

    rt = sub.add_parser(
        "retry",
        help="reprocess a meeting (transcription, diarization, and summary) that failed",
    )
    rt.add_argument("folder", help="meeting folder (name or path)")
    rt.add_argument("--wait", action="store_true",
                    help="process in foreground instead of background")
    rt.set_defaults(func=cmd_retry)

    im = sub.add_parser(
        "import",
        help="transcribe and summarize an external audio/video file (phone, etc.)",
    )
    im.add_argument("file", help="path to audio or video file to import")
    im.add_argument("--title", default="",
                    help="meeting title (default: filename)")
    im.add_argument("--lang", choices=["auto", "pt", "es", "en"], default="")
    im.add_argument("--output-lang", dest="output_lang",
                    choices=["meeting", "pt", "es", "en"], default="")
    im.add_argument("--wait", action="store_true",
                    help="process in foreground instead of background")
    im.set_defaults(func=cmd_import)

    pr = sub.add_parser("_process", help=argparse.SUPPRESS)
    pr.add_argument("path")
    pr.set_defaults(func=cmd_process)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # First run: if there's no config yet and the user didn't call 'setup'
    # or the internal worker, offer to run the interactive assistant now.
    if (
        not config_exists()
        and args.command not in ("setup", "_process")
        and sys.stdin.isatty()
    ):
        ui.log("first run: no config found.")
        _check_gpu_setup()
        try:
            response = input("run the configuration assistant now? [Y/n]: ").strip().lower()
        except EOFError:
            response = "n"
        if response in ("", "y", "yes"):
            rc = cmd_setup(args)
            if rc != 0:
                return rc
            print()
        else:
            print("using default values. Run 'notetaker setup' when you want to adjust.\n")

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
