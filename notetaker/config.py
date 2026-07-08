"""Loads and creates the Notetaker config (~/.config/notetaker/config.toml)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_PATH = Path.home() / ".config" / "notetaker" / "config.toml"

DEFAULT_CONFIG_TOML = """\
# Notetaker config. Empty device fields = auto-detection.
storage_root = "~/notetaker"

[audio]
# Empty = auto. Linux: uses the default PulseAudio source/sink. macOS: uses the
# avfoundation device index (mic = microphone index; monitor = loopback device
# index like BlackHole, required for online mode). Windows: uses the dshow
# device name (mic = microphone name; monitor = Stereo Mix name or virtual
# device like VB-CABLE, required for online mode).
mic_source = ""
monitor_source = ""

[whisper]
model = "medium"       # tiny | base | small | medium | large-v3
language = "auto"      # auto | pt | es | en
# NVIDIA GPU (cuda) is used automatically when detected; otherwise, CPU.

[summary]
# "meeting" = same language as the meeting. Or fix: pt | es | en
language = "meeting"

[llm]
# LLM Provider used to generate the summary from the transcription (via stdin).
provider = "kiro"      # kiro | claude
"""


@dataclass
class AudioConfig:
    mic_source: str = ""
    monitor_source: str = ""


@dataclass
class WhisperConfig:
    model: str = "medium"
    language: str = "auto"


@dataclass
class SummaryConfig:
    language: str = "meeting"


# Supported LLM providers and their corresponding CLI command. The command
# receives the transcription via stdin and returns the summary in Markdown via stdout.
LLM_PROVIDER_COMMANDS: dict[str, str] = {
    "kiro": "kiro-cli chat --no-interactive",
    "claude": "claude -p",
}


class InvalidLLMProviderError(ValueError):
    pass


def resolve_llm_command(provider: str) -> str:
    """Maps the LLM Provider (kiro|claude) to the actual CLI command."""
    try:
        return LLM_PROVIDER_COMMANDS[provider]
    except KeyError:
        options = ", ".join(LLM_PROVIDER_COMMANDS)
        raise InvalidLLMProviderError(
            f"Invalid LLM Provider: '{provider}'. Options: {options}."
        ) from None


@dataclass
class LLMConfig:
    provider: str = "kiro"


@dataclass
class Config:
    storage_root: Path = field(default_factory=lambda: Path.home() / "notetaker")
    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


def _expand(path_str: str) -> Path:
    return Path(path_str).expanduser()


def config_exists() -> bool:
    """Indicates whether the config has already been created (used to detect first run)."""
    return CONFIG_PATH.exists()


def ensure_config() -> Path:
    """Creates the config with defaults if it doesn't exist yet. Returns the path."""
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    return CONFIG_PATH


def _toml_str(value: str) -> str:
    """Escapes a string for TOML quoted value."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_config_toml(cfg: Config) -> str:
    """Renders a Config as commented TOML (same format as default)."""
    storage = str(cfg.storage_root)
    # Preserves the "~" when storage_root is within the user's HOME.
    home = str(Path.home())
    if storage == home:
        storage = "~"
    elif storage.startswith(home + "/"):
        storage = "~" + storage[len(home):]
    return f"""\
# Notetaker config. Empty device fields = auto-detection.
storage_root = {_toml_str(storage)}

[audio]
# Empty = auto. Linux: uses the default PulseAudio source/sink. macOS: uses the
# avfoundation device index (mic = microphone index; monitor = loopback device
# index like BlackHole, required for online mode). Windows: uses the dshow
# device name (mic = microphone name; monitor = Stereo Mix name or virtual
# device like VB-CABLE, required for online mode).
mic_source = {_toml_str(cfg.audio.mic_source)}
monitor_source = {_toml_str(cfg.audio.monitor_source)}

[whisper]
model = {_toml_str(cfg.whisper.model)}       # tiny | base | small | medium | large-v3
language = {_toml_str(cfg.whisper.language)}      # auto | pt | es | en
# NVIDIA GPU (cuda) is used automatically when detected; otherwise, CPU.

[summary]
# "meeting" = same language as the meeting. Or fix: pt | es | en
language = {_toml_str(cfg.summary.language)}

[llm]
# LLM Provider used to generate the summary from the transcription (via stdin).
provider = {_toml_str(cfg.llm.provider)}      # kiro | claude
"""


def write_config(cfg: Config) -> Path:
    """Writes the rendered config from a Config object. Returns the path."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(render_config_toml(cfg), encoding="utf-8")
    return CONFIG_PATH


def load_config() -> Config:
    """Loads the config, creating defaults on first run."""
    ensure_config()
    with CONFIG_PATH.open("rb") as fh:
        data = tomllib.load(fh)

    audio = data.get("audio", {})
    whisper = data.get("whisper", {})
    summary = data.get("summary", {})
    llm_data = data.get("llm", {})
    provider = llm_data.get("provider", "kiro")
    if provider not in LLM_PROVIDER_COMMANDS:
        options = ", ".join(LLM_PROVIDER_COMMANDS)
        raise InvalidLLMProviderError(
            f"Invalid LLM Provider in [llm].provider: '{provider}'. "
            f"Options: {options}."
        )

    return Config(
        storage_root=_expand(data.get("storage_root", "~/notetaker")),
        audio=AudioConfig(
            mic_source=audio.get("mic_source", ""),
            monitor_source=audio.get("monitor_source", ""),
        ),
        whisper=WhisperConfig(
            model=whisper.get("model", "medium"),
            language=whisper.get("language", "auto"),
        ),
        summary=SummaryConfig(
            language=summary.get("language", "meeting"),
        ),
        llm=LLMConfig(provider=provider),
    )
