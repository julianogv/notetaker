# Audio 100% local, only text goes to LLM Provider

Meeting audio is recorded and transcribed entirely on the user's machine
(faster-whisper). Only the text transcription is sent to the LLM Provider.
We decided this way because meetings can contain sensitive content and the
raw audio is the most critical data; the text goes to a contracted CLI with
data non-reuse policy.

Consequence: transcription runs on local CPU, without per-minute cost and offline,
at the cost of processing time. This is a privacy restriction that does not
appear in the code, so it is documented here.
