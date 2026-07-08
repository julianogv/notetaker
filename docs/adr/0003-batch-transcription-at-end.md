# Batch transcription at end, not live

Transcription runs in batch after `stop`, not in real time during a Meeting.
During recording we only capture audio (cheap); when ending, we transcribe the
files and generate the Summary. We decided this way because the target machine
is CPU-only, and Whisper (medium/large model) on CPU is too slow for real time
and would overload the notebook during the call.

Consequence: the user waits for processing after stop (running in background).
We gain maximum transcription quality and do not compete for CPU during the
meeting. Rejected alternative: live transcription with a small model, ruled out
due to poor quality in Portuguese and constant CPU usage.
