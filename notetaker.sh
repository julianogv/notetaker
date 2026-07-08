#!/usr/bin/env bash
#
# notetaker.sh — installs dependencies and runs Notetaker via uv.
#
# This script prepares everything Notetaker needs and runs it:
#   1. Installs 'uv' (Python package/venv manager) if absent.
#   2. Checks 'ffmpeg' (required to capture audio) and tries to install it.
#   3. Creates/syncs the virtual environment with 'uv sync'.
#   4. Runs Notetaker via 'uv run', passing through received arguments.
#
set -euo pipefail

# Project directory = directory of this script (works from anywhere).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --------------------------------------------------------------------------- #
# Colors (terminal only)
# --------------------------------------------------------------------------- #
if [ -t 1 ]; then
    C_BOLD="$(printf '\033[1m')"; C_DIM="$(printf '\033[2m')"
    C_GREEN="$(printf '\033[32m')"; C_YELLOW="$(printf '\033[33m')"
    C_RED="$(printf '\033[31m')"; C_RESET="$(printf '\033[0m')"
else
    C_BOLD=""; C_DIM=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_RESET=""
fi

ts()    { date '+%H:%M:%S'; }

info()  { printf '%s[%s]%s %s==>%s %s\n' "$C_DIM" "$(ts)" "$C_RESET" "$C_GREEN" "$C_RESET" "$*"; }
warn()  { printf '%s[%s]%s %s[warning]%s %s\n' "$C_DIM" "$(ts)" "$C_RESET" "$C_YELLOW" "$C_RESET" "$*" >&2; }
error() { printf '%s[%s]%s %s[error]%s %s\n' "$C_DIM" "$(ts)" "$C_RESET" "$C_RED" "$C_RESET" "$*" >&2; }

# --------------------------------------------------------------------------- #
# Help
# --------------------------------------------------------------------------- #
usage() {
    cat <<EOF
${C_BOLD}notetaker.sh${C_RESET} — records meetings, transcribes locally, and generates AI summaries.

${C_BOLD}USAGE${C_RESET}
    ./notetaker.sh <command> [options]
    ./notetaker.sh --setup        # only installs/updates dependencies
    ./notetaker.sh --help         # this help text

On first run, the script installs 'uv' (if necessary), checks 'ffmpeg',
and creates the virtual environment automatically. Then, all arguments
are passed through to Notetaker.

${C_BOLD}NOTETAKER COMMANDS${C_RESET}
    start "<title>"     Starts recording a meeting.
                        Live stream (time + size); Ctrl+C stops
                        and generates the summary.
    stop                Stops the current meeting and generates the summary.
    status              Shows the status of the most recent meeting.
    list                Lists all recorded meetings.
    devices             Shows detected audio devices.
    setup               Interactive setup wizard (language, model,
                        LLM, devices). Saves config.toml.
    summarize <folder>  Regenerates the summary from existing transcription.
    retry <folder>      Reprocesses a failed meeting (transcription,
                        diarization, and summary, from scratch, from the
                        recorded audio). Runs in background; use --wait to
                        watch in foreground.

${C_BOLD}'start' OPTIONS${C_RESET}
    --mode online|in-person|listener
                                 online = mic + system audio (default)
                                 in-person = mic only
                                 listener = system audio only
    --lang auto|pt|es|en         Language spoken in the meeting (default: config)
    --output-lang meeting|pt|es|en
                                 Summary language (default: meeting language)
    --diarization level1|level2  level1 = you vs. participants (default)
                                 level2 = identifies each speaker (ML, requires
                                          diarization add-on)
    --no-watch                   Does not live stream; use 'stop' later.

${C_BOLD}EXAMPLES${C_RESET}
    ./notetaker.sh --setup
    ./notetaker.sh start "sprint planning"
    ./notetaker.sh start "client meeting" --mode in-person
    ./notetaker.sh start "weekly" --lang en --output-lang pt
    ./notetaker.sh status
    ./notetaker.sh list
    ./notetaker.sh summarize 2026-07-02_1234_sprint-planning
    ./notetaker.sh retry 2026-07-02_1234_sprint-planning --wait

${C_BOLD}CONFIGURATION${C_RESET}
    Editable at: ~/.config/notetaker/config.toml
    (Whisper model, default language, LLM command, audio devices)
EOF
}

# --------------------------------------------------------------------------- #
# Ensures uv is installed
# --------------------------------------------------------------------------- #
ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi
    # Common locations where uv is installed if not in PATH.
    for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [ -x "$candidate" ]; then
            export PATH="$(dirname "$candidate"):$PATH"
            return
        fi
    done

    info "uv not found; installing..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        error "curl or wget is required to install uv. Please install one of them."
        exit 1
    fi

    # The installer places uv in ~/.local/bin (or ~/.cargo/bin).
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        error "uv installed but not found in PATH. Open a new terminal and try again."
        exit 1
    fi
    info "uv installed: $(uv --version)"
}

# --------------------------------------------------------------------------- #
# Checks/installs ffmpeg (system dependency for audio capture)
# --------------------------------------------------------------------------- #
ensure_ffmpeg() {
    if command -v ffmpeg >/dev/null 2>&1; then
        return
    fi
    warn "ffmpeg not found (required to record audio)."
    if command -v apt-get >/dev/null 2>&1; then
        info "attempting to install via apt-get (may ask for password)..."
        sudo apt-get update -qq && sudo apt-get install -y ffmpeg
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y ffmpeg
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --noconfirm ffmpeg
    elif command -v brew >/dev/null 2>&1; then
        brew install ffmpeg
    else
        error "Could not install ffmpeg automatically. Please install it manually."
        exit 1
    fi
}

# --------------------------------------------------------------------------- #
# Syncs the virtual environment (creates .venv and installs dependencies)
# --------------------------------------------------------------------------- #
sync_env() {
    info "syncing environment (uv sync)..."
    uv sync
}

# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #
main() {
    # Show help when explicitly requested or with no arguments.
    case "${1:-}" in
        -h|--help|help|"")
            usage
            exit 0
            ;;
    esac

    ensure_uv

    # Setup mode: only prepares environment and exits.
    if [ "${1:-}" = "--setup" ]; then
        ensure_ffmpeg
        sync_env
        info "ready. Use: ./notetaker.sh start \"my meeting\""
        exit 0
    fi

    # ffmpeg is only essential for recording (start); for other commands
    # we avoid blocking if it is not yet installed.
    if [ "${1:-}" = "start" ]; then
        ensure_ffmpeg
    fi

    # Ensures the environment is synced (fast when already up-to-date).
    sync_env

    info "running: notetaker $*"
    exec uv run notetaker "$@"
}

main "$@"
