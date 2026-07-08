# Level 1 diarization by default; level 2 (ML) optional

Recording always separates Tracks (mic and system) in online mode. Level 1
diarization uses this separation to distinguish you (mic) from participants
(system) without ML, at almost zero cost. Level 2, which identifies each speaker
via ML (whisperx/pyannote), is optional behind the `[diarization]` extra and is
not the default.

We decided this way because the target machine has no GPU: ML diarization is
heavy, unstable, and of uncertain value (generates "Speaker 1/2" that still
requires manual mapping). Level 1 covers the main case ("what I committed to"
vs "what they said"). Consequence: in in-person mode (mic only) there is no
separation at level 1. Level 2 can be adopted later without changing the
recording, since Tracks are always preserved.
