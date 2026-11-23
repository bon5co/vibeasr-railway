# VibeVoice-ASR-BitNet speech-to-text API for Railway

Microsoft's VibeVoice-ASR-BitNet (1.58-bit weights, MIT) run by Microsoft's VibeASR.cpp on CPU, behind an
OpenAI-style `POST /v1/audio/transcriptions` with a Bearer `API_KEY`. Weights are baked into the image.
Image: `ghcr.io/bon5co/vibeasr-railway`. See `NOTICE` for the licences of everything inside it.

- `GET /health` (no key): 503 until the models are loaded, then 200.
- `POST /v1/audio/transcriptions`: multipart `file` (any audio ffmpeg reads), optional `prompt` (hotwords),
  `response_format` = `json` | `text` | `verbose_json`.
- `GET /v1/models`: the model and the limits in force.

Environment: `API_KEY` (required), `MAX_UPLOAD_BYTES`, `MAX_AUDIO_SECONDS`, `PROCESS_TIMEOUT`.
Supported languages per the model card: en, zh, fr, it, ko, pt, vi. No timestamps (plain text only).
