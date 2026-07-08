"""Batch pipeline executed after stop: transcription -> diarization -> Summary.

Runs synchronously (the CLI dispatches it in background via detached subprocess),
updating meta.json at each phase to allow tracking via `status`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from . import diarize, transcribe
from .config import resolve_llm_command
from .prompts import resolve_output_language
from .storage import Meeting
from .summarize import generate_summary

# Progress callback: receives (phase, message). phase is a stable identifier
# ('transcribing_mic', 'transcribing_system', 'diarizing',
# 'summarizing', 'done'); message is human-readable text to display to the user.
ProgressCb = Callable[[str, str], None]


def _noop(phase: str, message: str) -> None:  # pragma: no cover
    pass


def process_meeting(meeting: Meeting, progress: ProgressCb | None = None) -> None:
    """Executes transcription, level 1/2 diarization, and Summary generation."""
    report = progress or _noop
    meta = meeting.read_meta()

    try:
        # --- Transcription of tracks ---
        meta.status = "transcribing"
        meeting.write_meta(meta)

        mic_t = None
        system_t = None
        timings: dict[str, dict] = {}

        device, _ = transcribe.detect_compute_device()
        report("device", f"  transcribing using: {device.upper()} (model {meta.whisper_model})")

        def _fmt(s: "transcribe.TranscribeStats") -> str:
            return (
                f"{s.audio_seconds:.0f}s of audio in {s.transcribe_seconds:.0f}s "
                f"(RTF {s.rtf:.2f}x"
                + (f", model loaded in {s.model_load_seconds:.0f}s" if s.model_load_seconds > 0.5 else "")
                + ")"
            )

        has_mic = meeting.audio_mic.exists()
        has_system = meeting.audio_system.exists()

        if has_mic and has_system:
            gpu = transcribe.gpu_available()
            if gpu:
                # GPU: CPU thread parallelization doesn't help and would duplicate
                # VRAM usage. Transcribe sequentially with cached model;
                # the GPU is already fast (low RTF).
                report("transcribing", "transcribing both tracks (GPU)...")
                mic_t = transcribe.transcribe_track(
                    meeting.audio_mic, meta.whisper_model, meta.lang
                )
                system_t = transcribe.transcribe_track(
                    meeting.audio_system, meta.whisper_model, meta.lang
                )
            else:
                # CPU: transcribe in parallel with separate models, each limited
                # to half the cores. CTranslate2 already saturates the cores,
                # so dividing threads avoids contention and reduces total time
                # (vs. sequential). Models are loaded sequentially first
                # (the loader is not thread-safe for concurrent use) and transcription
                # runs in threads.
                report("transcribing", "transcribing both tracks in parallel...")
                half = max(1, (os.cpu_count() or 2) // 2)
                model_mic = transcribe.load_model(meta.whisper_model, cpu_threads=half)
                model_system = transcribe.load_model(meta.whisper_model, cpu_threads=half)

                def _do(path, model):
                    return transcribe.transcribe_track(
                        path, meta.whisper_model, meta.lang, preloaded_model=model
                    )

                with ThreadPoolExecutor(max_workers=2) as ex:
                    fut_mic = ex.submit(_do, meeting.audio_mic, model_mic)
                    fut_system = ex.submit(_do, meeting.audio_system, model_system)
                    mic_t = fut_mic.result()
                    system_t = fut_system.result()

            meeting.transcript_mic.write_text(mic_t.text, encoding="utf-8")
            meeting.transcript_system.write_text(system_t.text, encoding="utf-8")
            timings["mic"] = vars(mic_t.stats)
            timings["system"] = vars(system_t.stats)
            report("stats_mic", "  mic: " + _fmt(mic_t.stats))
            report("stats_system", "  system: " + _fmt(system_t.stats))

        elif has_mic:
            report("transcribing_mic", "transcribing your speech (mic)...")
            mic_t = transcribe.transcribe_track(
                meeting.audio_mic, meta.whisper_model, meta.lang
            )
            meeting.transcript_mic.write_text(mic_t.text, encoding="utf-8")
            timings["mic"] = vars(mic_t.stats)
            report("stats_mic", "  mic: " + _fmt(mic_t.stats))

        elif has_system:
            report("transcribing_system", "transcribing the participants (system)...")
            system_t = transcribe.transcribe_track(
                meeting.audio_system, meta.whisper_model, meta.lang
            )
            meeting.transcript_system.write_text(system_t.text, encoding="utf-8")
            timings["system"] = vars(system_t.stats)
            report("stats_system", "  system: " + _fmt(system_t.stats))

        meta.extra["timings"] = timings

        primary = mic_t or system_t
        if primary is None:
            raise RuntimeError("no audio track found to transcribe")

        # Effective meeting language: language detected in mic track (or system).
        detected = primary.language
        meta.detected_lang = detected

        # --- Diarization ---
        report("diarizing", "organizing transcript by speaker...")
        if meta.mode == "import":
            # Single external source (mobile phone, video): no track separation, so
            # no level 1 diarization. Continuous transcript, without labels.
            full_text = diarize.render_plain(primary)
        else:
            if meta.diarization == "level2":
                diarize.build_level2(meeting.audio_mic, primary, detected)
            utterances = diarize.build_level1(mic_t, system_t, detected)
            full_text = diarize.render_transcript(utterances)
        meeting.transcript_full.write_text(full_text, encoding="utf-8")

        # --- Summary ---
        meta.status = "summarizing"
        meeting.write_meta(meta)

        report("summarizing", "generating summary with LLM...")
        out_lang = resolve_output_language(meta.output_lang, detected)
        md = generate_summary(
            full_text,
            meta.extra.get("llm_command", resolve_llm_command("kiro")),
            out_lang,
            title=meta.title,
        )
        meeting.summary_md.write_text(md, encoding="utf-8")

        meta.status = "done"
        meeting.write_meta(meta)
        report("done", "completed")

    except Exception as exc:  # noqa: BLE001
        meta.status = "error"
        meta.error = str(exc)
        meeting.write_meta(meta)
        raise
