"""CLI do Notetaker: start, stop, status, list, devices, summarize."""

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
    print(f"erro: {msg}", file=sys.stderr)
    return 1


def _audio_size(meeting: Meeting) -> int:
    """Soma o tamanho em bytes das Tracks de audio existentes."""
    total = 0
    for p in (meeting.audio_mic, meeting.audio_system):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def _run_processing(meeting: Meeting) -> int:
    """Roda o pipeline em foreground com spinner por fase."""
    from . import pipeline

    spinner = ui.Spinner("processando...").start()

    def on_progress(phase: str, message: str) -> None:
        if phase == "done":
            return
        # Fases 'stats_*' e 'device' sao diagnostico: imprime linha permanente.
        if phase.startswith("stats_") or phase == "device":
            spinner.stop()
            print(message)
            spinner.start()
            return
        spinner.update(message)

    try:
        pipeline.process_meeting(meeting, progress=on_progress)
    except Exception as exc:  # noqa: BLE001
        spinner.stop(f"falha no processamento: {exc}")
        return 1
    spinner.stop(f"resumo pronto: {meeting.resumo_md}")
    return 0


def _watch_recording(meeting: Meeting) -> None:
    """Monitor ao vivo: spinner + tempo decorrido + tamanho do audio.

    Bloqueia ate Ctrl+C. Nao encerra a gravacao aqui (o chamador trata).

    O tamanho/tempo vem dos logs do ffmpeg (read_progress): o muxer opus so
    grava os bytes no arquivo ao finalizar, entao o tamanho em disco fica 0
    durante a captura. O log reporta o progresso corrente.
    """
    logs = [meeting.ffmpeg_log_mic, meeting.ffmpeg_log_system]
    frames = ui._frames()
    i = 0
    while True:
        frame = frames[i % len(frames)]
        size, elapsed = audio.read_progress(logs)
        ui.status_line(
            f"{frame} gravando  {ui.format_duration(elapsed)}  "
            f"audio: {ui.format_size(size)}  (Ctrl+C para encerrar)"
        )
        i += 1
        time.sleep(0.2)


def _dispatch_background_processing(meeting: Meeting) -> None:
    """Dispara o pipeline de processamento em background, desanexado do shell."""
    subprocess.Popen(
        [sys.executable, "-m", "notetaker.cli", "_process", str(meeting.path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **audio.detached_worker_kwargs(),
    )


def _finalize_stopped_meta(meeting: Meeting) -> Meta:
    """Encerra o ffmpeg e marca o meta como 'transcribing'. Retorna o meta."""
    meta = meeting.read_meta()
    audio.stop_recording(meta.ffmpeg_pids)  # aguarda o ffmpeg finalizar o opus
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
    """Pergunta o que fazer quando ja ha uma reuniao gravando.

    Retorna 'stop' (encerrar a atual e iniciar a nova), 'stop_only' (apenas
    encerrar a atual, sem iniciar nova), 'new' (iniciar numa nova pasta, deixando
    a atual gravando) ou 'abort' (cancelar). So e chamada quando ha terminal
    interativo.
    """
    print(f"ja existe uma reuniao gravando: {active.path.name}.")
    print("  [e] encerrar a gravacao em andamento e iniciar a nova")
    print("  [s] apenas encerrar a gravacao em andamento (nao inicia nova)")
    print("  [n] iniciar em uma nova pasta (mantem a atual gravando)")
    print("  [c] cancelar")
    while True:
        try:
            resp = input("o que deseja fazer? [e/s/n/c]: ").strip().lower()
        except EOFError:
            return "abort"
        if resp in ("e", "encerrar"):
            return "stop"
        if resp in ("s", "so", "apenas"):
            return "stop_only"
        if resp in ("n", "nova", "novo"):
            return "new"
        if resp in ("c", "cancelar", ""):
            return "abort"
        print("  opcao invalida. Escolha e, s, n ou c.")


# --------------------------------------------------------------------------- #
# start
# --------------------------------------------------------------------------- #
def cmd_start(args: argparse.Namespace) -> int:
    cfg = load_config()

    active = find_active_meeting(cfg.storage_root)
    if active:
        # Sem terminal interativo (script, pipe, --no-watch encadeado) nao da
        # para perguntar: mantem o comportamento seguro de nao mexer na
        # gravacao em andamento.
        if not sys.stdin.isatty():
            return _err(
                f"ja existe uma reuniao gravando: {active.path.name}. "
                "Rode 'notetaker stop'."
            )
        decision = _prompt_active_conflict(active)
        if decision == "abort":
            print("cancelado; a gravacao em andamento foi mantida.")
            return 1
        if decision == "stop_only":
            print(f"encerrando a reuniao em andamento: {active.path.name}...")
            _finalize_stopped_meta(active)
            _dispatch_background_processing(active)
            print("reuniao encerrada; processando em background. "
                  "Acompanhe com 'notetaker status'.")
            return 0
        if decision == "stop":
            print(f"encerrando a reuniao em andamento: {active.path.name}...")
            _finalize_stopped_meta(active)
            _dispatch_background_processing(active)
            print("reuniao anterior encerrada; processando em background.")
        # decision == "new": segue adiante e cria uma nova pasta (a atual
        # continua gravando; o proximo 'stop' encerra a mais recente).

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

    print(f"gravando: {meeting.path.name}")
    print(f"  modo: {args.mode}")
    if devices.mic_source:
        print(f"  mic: {devices.mic_source}")
    if devices.monitor_source:
        print(f"  system: {devices.monitor_source}")
    print("  (Ctrl+C encerra a gravacao e gera o resumo)")

    # Modo detached: retorna e deixa o usuario encerrar com 'notetaker stop'.
    if not args.watch:
        print("rode 'notetaker stop' para encerrar e gerar o resumo.")
        return 0

    # Modo watch (padrao): monitor ao vivo ate Ctrl+C, depois processa.
    try:
        _watch_recording(meeting)
    except KeyboardInterrupt:
        pass

    ui.clear_line()
    print("\nencerrando a gravacao...")
    return _finish_recording(meeting)


def _finish_recording(meeting: Meeting) -> int:
    """Encerra o ffmpeg, atualiza meta e dispara o processamento em foreground."""
    meta = meeting.read_meta()

    # Encerra o ffmpeg com feedback (a finalizacao do container opus pode levar
    # alguns segundos em gravacoes longas).
    spinner = ui.Spinner("finalizando os arquivos de audio...").start()
    audio.stop_recording(meta.ffmpeg_pids)  # aguarda o ffmpeg finalizar o opus
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

    print(f"gravacao encerrada: {meeting.path.name} "
          f"({ui.format_duration(meta.duration_seconds)}, "
          f"{ui.format_size(_audio_size(meeting))})")
    print("iniciando transcricao local e geracao do resumo...")
    return _run_processing(meeting)


# --------------------------------------------------------------------------- #
# stop
# --------------------------------------------------------------------------- #
def cmd_stop(args: argparse.Namespace) -> int:
    cfg = load_config()
    meeting = find_active_meeting(cfg.storage_root)
    if meeting is None:
        return _err("nenhuma reuniao em gravacao.")

    _finalize_stopped_meta(meeting)

    print(f"gravacao encerrada: {meeting.path.name}")

    if args.wait:
        return _run_processing(meeting)

    _dispatch_background_processing(meeting)
    print("processando em background. Acompanhe com 'notetaker status'.")
    return 0


# --------------------------------------------------------------------------- #
# _process (interno, chamado em background pelo stop)
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
        print("nenhuma reuniao encontrada.")
        return 0

    latest = meetings[0]
    meta = latest.read_meta()
    print(f"reuniao: {latest.path.name}")
    print(f"  status: {meta.status}")
    print(f"  modo: {meta.mode} | diarizacao: {meta.diarization}")
    if meta.detected_lang:
        print(f"  idioma detectado: {meta.detected_lang}")
    if meta.error:
        print(f"  erro: {meta.error}")
    if meta.status == "done":
        print(f"  resumo: {latest.resumo_md}")
    return 0


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #
def cmd_list(args: argparse.Namespace) -> int:
    cfg = load_config()
    meetings = list_meetings(cfg.storage_root)
    if not meetings:
        print("nenhuma reuniao encontrada.")
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
# summarize (regenera a partir da transcricao existente)
# --------------------------------------------------------------------------- #
def cmd_summarize(args: argparse.Namespace) -> int:
    cfg = load_config()
    meeting = resolve_meeting(cfg.storage_root, args.pasta)
    if meeting is None:
        return _err(f"reuniao nao encontrada: {args.pasta}")
    if not meeting.transcript_full.exists():
        return _err("transcript-full.txt nao encontrada; rode o pipeline primeiro.")

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

    meeting.resumo_md.write_text(md, encoding="utf-8")
    print(f"resumo regenerado: {meeting.resumo_md}")
    return 0


# --------------------------------------------------------------------------- #
# retry (reprocessa desde a transcricao apos uma falha)
# --------------------------------------------------------------------------- #
def cmd_retry(args: argparse.Namespace) -> int:
    cfg = load_config()
    meeting = resolve_meeting(cfg.storage_root, args.pasta)
    if meeting is None:
        return _err(f"reuniao nao encontrada: {args.pasta}")
    if not meeting.audio_mic.exists() and not meeting.audio_system.exists():
        return _err("nenhuma Track de audio encontrada; nao ha o que reprocessar.")

    meta = meeting.read_meta()
    meta.error = ""
    meta.status = "transcribing"
    meeting.write_meta(meta)

    print(f"reprocessando: {meeting.path.name}")

    if args.wait:
        return _run_processing(meeting)

    _dispatch_background_processing(meeting)
    print("processando em background. Acompanhe com 'notetaker status'.")
    return 0


# --------------------------------------------------------------------------- #
# import (transcreve e resume um unico arquivo de audio/video externo)
# --------------------------------------------------------------------------- #
def cmd_import(args: argparse.Namespace) -> int:
    """Importa um arquivo externo (audio ou video), transcreve e gera o resumo.

    A fonte pode ter sido gravada em outro lugar (celular, gravador, video de
    call). O audio e extraido/convertido para o formato das Tracks (opus mono)
    e a Meeting e processada pelo mesmo pipeline batch. Como ha uma unica fonte,
    nao ha separacao de locutor por Track: o modo 'import' gera transcricao
    corrida (sem rotulos).
    """
    cfg = load_config()

    src = Path(args.arquivo).expanduser()
    if not src.exists():
        return _err(f"arquivo nao encontrado: {src}")
    if not src.is_file():
        return _err(f"nao e um arquivo: {src}")

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

    # Extrai o audio para a Track mic (fonte unica). Descarta video se houver.
    spinner = ui.Spinner(f"extraindo o audio de {src.name}...").start()
    try:
        audio.import_audio(src, meeting.audio_mic)
    except audio.AudioError as exc:
        spinner.stop()
        meta.status = "error"
        meta.error = str(exc)
        meeting.write_meta(meta)
        return _err(str(exc))
    spinner.stop(f"audio importado: {meeting.path.name}")

    print("iniciando transcricao local e geracao do resumo...")

    if args.wait:
        return _run_processing(meeting)

    _dispatch_background_processing(meeting)
    print("processando em background. Acompanhe com 'notetaker status'.")
    return 0


# --------------------------------------------------------------------------- #
# setup (assistente interativo de configuracao)
# --------------------------------------------------------------------------- #
def _prompt(label: str, default: str, choices: list[str] | None = None) -> str:
    """Le uma resposta do usuario com valor padrao.

    Enter aceita o default. Quando ha 'choices', repete ate a resposta ser
    valida (case-insensitive). Vazio e uma resposta valida (mantem o default).
    """
    hint = f" [{'/'.join(choices)}]" if choices else ""
    suffix = f" (padrao: {default})" if default else " (padrao: vazio = auto)"
    while True:
        try:
            resposta = input(f"{label}{hint}{suffix}: ").strip()
        except EOFError:
            return default
        if not resposta:
            return default
        if choices and resposta.lower() not in [c.lower() for c in choices]:
            print(f"  opcao invalida. Escolha uma de: {', '.join(choices)}")
            continue
        return resposta


def _check_gpu_setup() -> None:
    """Na primeira execucao, avisa sobre a lib CUDA quando ha GPU NVIDIA.

    O CTranslate2 (backend do faster-whisper) so acelera em GPUs NVIDIA e
    depende do libcublas. Se o hardware tem GPU mas as libs ainda nao estao
    utilizaveis, orienta a instalacao para aproveitar a aceleracao.
    """
    if not transcribe.nvidia_gpu_present():
        return
    if transcribe.gpu_available():
        # GPU ja utilizavel (libs presentes): nada a fazer.
        print("GPU NVIDIA detectada e pronta para acelerar a transcricao.\n")
        return

    print("\nGPU NVIDIA detectada, mas a lib CUDA (libcublas) nao esta pronta.")
    print("Instale-a para acelerar a transcricao:")
    print("  sudo apt-get install -y libcublas-12-0\n")


def cmd_setup(args: argparse.Namespace) -> int:
    """Assistente interativo: pergunta cada opcao com o valor padrao e grava o config."""
    # Parte de um config existente (se houver) para preservar valores atuais.
    base = load_config() if config_exists() else Config()

    print("configuracao do Notetaker")
    print(f"o config sera gravado em: {CONFIG_PATH}")
    print("pressione Enter para aceitar o valor padrao entre parenteses.\n")

    storage = _prompt("pasta das reunioes (storage_root)", str(base.storage_root))

    print("\n-- audio (deixe vazio para deteccao automatica no 'start') --")
    mic_source = _prompt("dispositivo do microfone (mic_source)", base.audio.mic_source)
    monitor_source = _prompt(
        "dispositivo do audio do sistema (monitor_source)", base.audio.monitor_source
    )

    print("\n-- transcricao (whisper) --")
    model = _prompt(
        "modelo Whisper", base.whisper.model,
        choices=["tiny", "base", "small", "medium", "large-v3"],
    )
    language = _prompt(
        "idioma falado nas reunioes", base.whisper.language,
        choices=["auto", "pt", "es", "en"],
    )

    print("\n-- resumo --")
    summary_language = _prompt(
        "idioma do resumo", base.summary.language,
        choices=["meeting", "pt", "es", "en"],
    )

    print("\n-- LLM (CLI que recebe a transcricao via stdin) --")
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
    print(f"\nconfig gravado em {path}")
    print("pronto. Use: notetaker start \"minha reuniao\"")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="notetaker", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("start", help="inicia a gravacao de uma reuniao")
    s.add_argument("title", help="titulo da reuniao")
    s.add_argument(
        "--mode", choices=["online", "presencial", "listener"], default="online"
    )
    s.add_argument("--lang", choices=["auto", "pt", "es", "en"], default="")
    s.add_argument("--diarization", choices=["level1", "level2"], default="level1")
    s.add_argument("--output-lang", dest="output_lang",
                   choices=["meeting", "pt", "es", "en"], default="")
    s.add_argument("--no-watch", dest="watch", action="store_false",
                   help="nao acompanha ao vivo; retorna e aguarda 'notetaker stop'")
    s.set_defaults(func=cmd_start, watch=True)

    st = sub.add_parser("stop", help="encerra a gravacao e gera o resumo")
    st.add_argument("--wait", action="store_true",
                    help="processa em primeiro plano em vez de background")
    st.set_defaults(func=cmd_stop)

    stt = sub.add_parser("status", help="mostra o estado da reuniao mais recente")
    stt.set_defaults(func=cmd_status)

    ls = sub.add_parser("list", help="lista reunioes")
    ls.set_defaults(func=cmd_list)

    dv = sub.add_parser("devices", help="mostra os dispositivos de audio detectados")
    dv.set_defaults(func=cmd_devices)

    su = sub.add_parser("setup", help="assistente interativo de configuracao")
    su.set_defaults(func=cmd_setup)

    sm = sub.add_parser("summarize", help="regenera o resumo a partir da transcricao")
    sm.add_argument("pasta", help="pasta da reuniao (nome ou caminho)")
    sm.add_argument("--output-lang", dest="output_lang",
                    choices=["meeting", "pt", "es", "en"], default="")
    sm.set_defaults(func=cmd_summarize)

    rt = sub.add_parser(
        "retry",
        help="reprocessa uma reuniao (transcricao, diarizacao e resumo) que falhou",
    )
    rt.add_argument("pasta", help="pasta da reuniao (nome ou caminho)")
    rt.add_argument("--wait", action="store_true",
                    help="processa em primeiro plano em vez de background")
    rt.set_defaults(func=cmd_retry)

    im = sub.add_parser(
        "import",
        help="transcreve e resume um arquivo de audio/video externo (celular, etc.)",
    )
    im.add_argument("arquivo", help="caminho do arquivo de audio ou video a importar")
    im.add_argument("--title", default="",
                    help="titulo da reuniao (padrao: nome do arquivo)")
    im.add_argument("--lang", choices=["auto", "pt", "es", "en"], default="")
    im.add_argument("--output-lang", dest="output_lang",
                    choices=["meeting", "pt", "es", "en"], default="")
    im.add_argument("--wait", action="store_true",
                    help="processa em primeiro plano em vez de background")
    im.set_defaults(func=cmd_import)

    pr = sub.add_parser("_process", help=argparse.SUPPRESS)
    pr.add_argument("path")
    pr.set_defaults(func=cmd_process)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Primeira execucao: se ainda nao ha config e o usuario nao chamou 'setup'
    # nem o worker interno, oferece rodar o assistente interativo agora.
    if (
        not config_exists()
        and args.command not in ("setup", "_process")
        and sys.stdin.isatty()
    ):
        print("primeira execucao: nenhum config encontrado.")
        _check_gpu_setup()
        try:
            resposta = input("rodar o assistente de configuracao agora? [S/n]: ").strip().lower()
        except EOFError:
            resposta = "n"
        if resposta in ("", "s", "sim", "y", "yes"):
            rc = cmd_setup(args)
            if rc != 0:
                return rc
            print()
        else:
            print("usando valores padrao. Rode 'notetaker setup' quando quiser ajustar.\n")

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
