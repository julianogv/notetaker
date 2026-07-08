# LLM Provider via external CLI, not direct API

Notetaker generates the Summary by invoking an already-contracted LLM CLI (kiro-cli,
claude-code, etc.) via a configurable command template, sending the transcript
via stdin and receiving the summary in Markdown via stdout. We decided this way to
reuse existing signatures with data non-reuse policy, avoid managing API keys,
and decouple from the provider.

Trade-off: we depend on the CLI accepting stdin input and one-shot mode, and
we need to extract Markdown from stdout (which may come with banners/ANSI/extra text)
instead of a structured API response. For this, the LLM delimits the summary
between sentinels. Rejected alternative: call LLM APIs directly, which would bring
credential management and provider coupling. We also considered asking for JSON
from the LLM, but Markdown is more robust to extract from CLI noise and is already
the desired final format.
