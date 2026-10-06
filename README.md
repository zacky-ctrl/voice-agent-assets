# Voice agent audio assets

Background ambience for the Smile Studio dental receptionist voice agent (Vapi).

- `clinic-ambience.mp3` — 3-minute seamless loop: room tone, ceiling fan, unintelligible distant waiting-room voices, and sparse irregular desk sounds (typing, mouse, paper, chair, distant horn). Band-limited to telephone range (300–3400 Hz), mixed at about -42 LUFS so it sits well below speech.
- `make_ambience.py` — script that generates it. All sounds are synthesized or made with system text-to-speech, so there are no licensing issues.
