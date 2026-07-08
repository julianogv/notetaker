"""LLM Provider: invokes an external CLI, sending the prompt via stdin.

The command is resolved from the LLM Provider chosen in config
([llm].provider: "kiro" or "claude") via `config.resolve_llm_command`.
The transcript + instructions go via stdin (avoids argument limit). The response
comes via stdout and is returned raw for summarize to parse.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess


class LLMError(RuntimeError):
    pass


def run_llm(command: str, prompt: str, timeout: int = 600) -> str:
    """Executes the LLM Provider passing `prompt` via stdin and returns stdout.

    `command` is a shell string (e.g., "claude -p"); it's split with shlex
    to avoid shell interpretation of the transcript content. On Windows we use
    posix=False so that backslashes in paths (e.g., C:\\Tools\\llm.exe) are not
    interpreted as escape characters.
    """
    argv = shlex.split(command, posix=(os.name != "nt"))
    if not argv:
        raise LLMError("empty LLM command (resolved from [llm].provider)")

    if shutil.which(argv[0]) is None:
        raise LLMError(
            f"LLM CLI not found: '{argv[0]}'. "
            f"Adjust [llm].provider in config (kiro | claude) or install the CLI."
        )

    try:
        proc = subprocess.run(
            argv,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise LLMError(f"LLM Provider exceeded time limit ({timeout}s)") from exc

    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise LLMError(
            f"LLM Provider returned code {proc.returncode}: {detail}"
        )

    output = proc.stdout.strip()
    if not output:
        raise LLMError("LLM Provider returned no output")
    return output
