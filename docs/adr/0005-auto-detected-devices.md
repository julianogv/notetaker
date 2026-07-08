# Audio devices auto-detected at start, not fixed in config

The mic and output monitor are resolved at runtime on `start`, from the current
PulseAudio default source/sink (`pactl get-default-source` /
`get-default-sink`, deriving the `.monitor` from the sink). The config offers
only optional overrides; empty means auto-detection.

The resolution is per platform backend, maintaining the same auto-detection
semantics:
- **Linux (PulseAudio/PipeWire)**: default source/sink; monitor = `<sink>.monitor`.
- **macOS (AVFoundation)**: device indices; monitor = index of a loopback device
  (BlackHole), required for online mode.
- **Windows (DirectShow)**: device names; monitor = name of "Stereo Mix"
  or a virtual device (VB-CABLE/VoiceMeeter), required for online mode.

We decided this way because the tool will be distributed to other people and
users switch headsets/speakers frequently. Fixing device names in config would
break with each switch and make distribution impractical. Consequence: capture
uses the active device at the time of start; to force a specific device, the
user fills in the override in config.
