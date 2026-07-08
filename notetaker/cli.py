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
        spinner.stop(f"processing failed: {exc}")
        return 1
    spinner.stop(f"summary ready: {meeting.summary_md}")
    return 0


def _watch_recording(meeting: Meeting) -> None:
    """Live monitoring: spinner + elapsed time + audio size.

    Blocks until Ctrl+C. Does not stop the recording here (caller handles it).

    Size/time come from ffmpeg logs (read_progress): the opus muxer only
    writes bytes to the file on finalization, so disk size stays 0 during
    capture. The log reports current progress.
    """
    logs = [meeting.ffmpeg_log_mic, meeting.ffmpeg_log_system]
    frames = ui._frames()
    i = 0
    while True:
        frame = frames[i % len(frames)]
        size, elapsed = audio.read_progress(logs)
        ui.status_line(
            f"{frame} recording  {ui.format_duration(elapsed)}  "
            f"audio: {ui.format_size(size)}  (Ctrl+C to stop)"
        )
        i += 1
        time.sleep(0.2)


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


def _prompt_active_conflict(active: Meeting) -> str:
    """Ask what to do when a meeting is already recording.

    Returns 'stop' (stop current and start new), 'stop_only' (stop current only,
    don't start new), 'new' (start in new folder, keep current recording), or
    'abort' (cancel). Only called when there's an interactive terminal.
    """
    print(f"a meeting is already recording: {active.path.name}.")
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
        # Without interactive terminal (script, pipe, --no-watch chained) cannot
        # ask: maintain safe behavior of not touching the running recording.
        if not sys.stdin.isatty():
            return _err(
                f"a meeting is already recording: {active.path.name}. "
                "Run 'notetaker stop'."
            )
        decision = _prompt_active_conflict(active)
        if decision == "abort":
            print("cancelled; the running recording was kept.")
            return 1
        if decision == "stop_only":
            print(f"stopping current meeting: {active.path.name}...")
            _finalize_stopped_meta(active)
            _dispatch_background_processing(active)
            print("meeting stopped; processing in background. "
                  "Check with 'notetaker status'.")
            return 0
        if decision == "stop":
            print(f"stopping current meeting: {active.path.name}...")
            _finalize_stopped_meta(active)
            _dispatch_background_processing(active)
            print("previous meeting stopped; processing in background.")
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
        extra={"llm_command": resolve_llm_command(cfg.llm.provider)},
    )

    try:
        pids = audio.start_recording(meeting, devices, args.mode)
    except audio.AudioError as exc:
        meta.status = "error"
        meta.error = str(exc)
        meeting.write_meta(meta)
        return _err(str(exc))

    meta.ffmpeg_pids = pids
    meeting.write_meta(meta)

    print(f"recording: {meeting.path.name}")
    print(f"  mode: {args.mode}")
    if devices.mic_source:
        print(f"  mic: {devices.mic_source}")
    if devices.monitor_source:
        print(f"  system: {devices.monitor_source}")
    print("  (Ctrl+C stops recording and generates summary)")

    # Detached mode: return and let user stop with 'notetaker stop'.
    if not args.watch:
        print("run 'notetaker stop' to stop recording and generate the summary.")
        return 0

    # Watch mode (default): live monitoring until Ctrl+C, then process.
    try:
        _watch_recording(meeting)
    except KeyboardInterrupt:
        pass

    ui.clear_line()
    print("\nstopping recording...")
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

    print(f"recording stopped: {meeting.path.name} "
          f"({ui.format_duration(meta.duration_seconds)}, "
          f"{ui.format_size(_audio_size(meeting))})")
    print("starting local transcription and summary generation...")
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

    print(f"recording stopped: {meeting.path.name}")

    if args.wait:
        return _run_processing(meeting)

    _dispatch_background_processing(meeting)
    print("processing in background. Check with 'notetaker status'.")
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
# list
# --------------------------------------------------------------------------- #
def cmd_list(args: argparse.Namespace) -> int:
    cfg = load_config()
    meetings = list_meetings(cfg.storage_root)
    if not meetings:
        print("no meetings found.")
        return 0
    for m in meetings:
        meta = m.read_meta()
        print(f"{m.path.name:50s} [{meta.status}]")
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
    if not meeting.transcript_full.exists():
        return _err("transcript-full.txt not found; run the pipeline first.")

    from .summarize import generate_summary

    meta = meeting.read_meta()
    transcript = meeting.transcript_full.read_text(encoding="utf-8")
    out_lang = resolve_output_language(
        args.output_lang or meta.output_lang, meta.detected_lang or meta.lang
    )
    llm_command = meta.extra.get("llm_command", resolve_llm_command(cfg.llm.provider))

    try:
        md = generate_summary(transcript, llm_command, out_lang, title=meta.title)
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))

    meeting.summary_md.write_text(md, encoding="utf-8")
    print(f"summary regenerated: {meeting.summary_md}")
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

    print(f"reprocessing: {meeting.path.name}")

    if args.wait:
        return _run_processing(meeting)

    _dispatch_background_processing(meeting)
    print("processing in background. Check with 'notetaker status'.")
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
    spinner.stop(f"audio imported: {meeting.path.name}")

    print("starting local transcription and summary generation...")

    if args.wait:
        return _run_processing(meeting)

    _dispatch_background_processing(meeting)
    print("processing in background. Check with 'notetaker status'.")
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

    cfg = Config(
        storage_root=Path(storage).expanduser(),
        audio=type(base.audio)(mic_source=mic_source, monitor_source=monitor_source),
        whisper=type(base.whisper)(model=model, language=language),
        summary=type(base.summary)(language=summary_language),
        llm=type(base.llm)(provider=llm_provider),
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

    ls = sub.add_parser("list", help="list meetings")
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
        print("first run: no config found.")
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
