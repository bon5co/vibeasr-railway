#!/usr/bin/env python3
"""VibeVoice-ASR-BitNet speech-to-text API on CPU, for Railway.

Microsoft's VibeVoice-ASR-BitNet (1.58-bit weights, MIT) run by Microsoft's VibeASR.cpp ggml runtime,
behind an OpenAI-style transcription endpoint. Send audio, get text back. What this file adds around the
runtime, because it ships no HTTP server:

  1. Bearer API_KEY on every route except GET /health. An empty API_KEY refuses to start.
  2. GET /health answers 503 until the models are loaded, then 200, unauthenticated.
  3. POST /v1/audio/transcriptions (multipart: file, model, prompt, response_format) in the shape of the
     OpenAI route. response_format is json (default), text or verbose_json; verbose_json adds the audio
     duration and an empty segments list, because this model returns plain text with no timestamps (its
     "json" prompt mode was tried on the shipped weights and answers one token). `prompt` is passed to the
     model as hotwords (names, jargon) in the model's own "extra info" slot.
  4. GET /v1/models lists the model and the limits in force.
  5. Any audio ffmpeg can read is accepted; it is converted to mono 24 kHz WAV, the rate the model uses.
     Upload size and audio length are capped. Audio is held in a per-request temp dir that is deleted
     afterwards and is never logged.
  6. Weights are baked into the image: nothing downloads at boot. One persistent asr_stream_server process
     keeps them loaded; requests run one at a time behind a lock, and a request that arrives while one is
     running gets 429 with Retry-After instead of queueing past Railway's ~5 minute edge timeout.
"""

import asyncio
import hmac
import os
import re
import socket
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

API_KEY = os.environ.get("API_KEY") or ""
THREADS = int(os.environ.get("ASR_THREADS") or os.cpu_count() or 1)
MODEL_DIR = os.environ.get("MODEL_DIR") or "/models"
ASR_BIN = os.environ.get("ASR_BIN") or "/usr/local/bin/asr_stream_server"
LM_MODEL = os.path.join(MODEL_DIR, "vibeasr-lm-i2_s-embed-q6_k.gguf")
VAE_MODEL = os.path.join(MODEL_DIR, "vibeasr-vae-encoder-i8_s.gguf")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES") or 100 * 1024 * 1024)
MAX_AUDIO_SECONDS = float(os.environ.get("MAX_AUDIO_SECONDS") or 600)
PROCESS_TIMEOUT = float(os.environ.get("PROCESS_TIMEOUT") or 270)
MODEL_ID = "vibevoice-asr-bitnet"
PUBLIC_PATHS = {"/health"}
SAMPLE_RATE = 24000

app = FastAPI(
    title="VibeVoice-ASR-BitNet speech-to-text API",
    version="1.0.0",
    description="Microsoft VibeVoice-ASR-BitNet (MIT) on CPU. OpenAI-style /v1/audio/transcriptions. Authorization: Bearer <API_KEY>.",
)

_lock = threading.Lock()          # one transcription at a time
_proc = None                       # asr_stream_server subprocess
_ready = threading.Event()
_start_lock = threading.Lock()
_last_rtf = None
_start_error = None


def error(status, message, kind="invalid_request_error", headers=None):
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status, headers=headers)


def _pump_stderr(proc):
    global _last_rtf
    for line in proc.stderr:
        line = line.rstrip()
        m = re.search(r"RTF: ([0-9.]+)", line)
        if m:
            _last_rtf = float(m.group(1))
        print("[asr]", line, flush=True)


def start_server():
    """(Re)start the persistent runtime and wait for its ---READY--- line. Serialised, and a no-op when up."""
    with _start_lock:
        if _ready.is_set() and _proc is not None and _proc.poll() is None:
            return True
        return _start_server()


def _start_server():
    global _proc, _start_error
    _ready.clear()
    cmd = [ASR_BIN, "--vae-model", VAE_MODEL, "--lm-model", LM_MODEL, "-t", str(THREADS),
           "--greedy", "--no-token-stream"]
    _proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, bufsize=1)
    threading.Thread(target=_pump_stderr, args=(_proc,), daemon=True).start()
    t0 = time.time()
    while True:
        line = _proc.stdout.readline()
        if not line:
            _start_error = f"asr_stream_server exited with {_proc.poll()} before ready"
            print("[app]", _start_error, flush=True)
            return False
        if line.strip() == "---READY---":
            _start_error = None
            _ready.set()
            print(f"[app] runtime ready in {time.time() - t0:.1f}s, threads={THREADS}", flush=True)
            return True


def _send(line):
    _proc.stdin.write(line + "\n")
    _proc.stdin.flush()


def _read_until(marker):
    out = []
    while True:
        line = _proc.stdout.readline()
        if not line:
            raise RuntimeError("runtime exited mid-request")
        if line.strip() == marker:
            return "".join(out)
        out.append(line)


def run_model(wav_path, prompt_format, context):
    """One transcription on the persistent runtime. Caller holds _lock."""
    if not start_server():
        raise RuntimeError(_start_error)
    _send("CONTEXT:" + context)
    _read_until("---ACK---")
    _send("FORMAT:" + prompt_format)
    _read_until("---ACK---")
    _send(wav_path)
    return _read_until("---END---").strip()


def kill_runtime():
    global _proc
    _ready.clear()
    if _proc is not None and _proc.poll() is None:
        _proc.kill()
    _proc = None


@app.middleware("http")
async def auth(request: Request, call_next):
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)
    header = request.headers.get("authorization") or ""
    token = header[7:] if header.lower().startswith("bearer ") else ""
    if not token or not hmac.compare_digest(token.encode(), API_KEY.encode()):
        return error(401, "Missing or invalid API key. Send Authorization: Bearer <API_KEY>.",
                     "authentication_error", {"WWW-Authenticate": "Bearer"})
    return await call_next(request)


@app.get("/health")
def health():
    if _ready.is_set() and _proc is not None and _proc.poll() is None:
        return {"status": "ok", "model": MODEL_ID, "threads": THREADS}
    return JSONResponse({"status": "loading", "error": _start_error}, status_code=503)


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{
        "id": MODEL_ID, "object": "model", "owned_by": "microsoft",
        "languages": ["en", "zh", "fr", "it", "ko", "pt", "vi"],
        "limits": {"max_upload_bytes": MAX_UPLOAD_BYTES, "max_audio_seconds": MAX_AUDIO_SECONDS,
                   "process_timeout_seconds": PROCESS_TIMEOUT},
        "last_realtime_factor": _last_rtf,
    }]}


def convert(src, dst):
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-vn", "-ac", "1",
                        "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", dst],
                       capture_output=True, text=True, timeout=120)
    return r.returncode == 0 and os.path.getsize(dst) > 44


@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile = File(...), model: str = Form(MODEL_ID), prompt: str = Form(""),
                     response_format: str = Form("json"), language: str = Form(""),
                     temperature: float = Form(0.0)):
    if response_format not in ("json", "text", "verbose_json"):
        return error(400, "response_format must be json, text or verbose_json.")
    if "\n" in prompt or "\r" in prompt or len(prompt) > 2000:
        return error(400, "prompt is hotwords on one line, at most 2000 characters.")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        return error(413, f"Upload larger than {MAX_UPLOAD_BYTES} bytes.")
    if not data:
        return error(400, "Empty file.")
    if not _lock.acquire(blocking=False):
        return error(429, "Another transcription is running; this service handles one at a time.",
                     "rate_limit_error", {"Retry-After": "30"})
    tmp = tempfile.mkdtemp(prefix="asr-")
    try:
        src, wav = os.path.join(tmp, "in"), os.path.join(tmp, "audio.wav")
        with open(src, "wb") as f:
            f.write(data)
        del data
        if not await asyncio.to_thread(convert, src, wav):
            return error(400, "Could not decode the file as audio (ffmpeg refused it).")
        duration = (os.path.getsize(wav) - 44) / (SAMPLE_RATE * 2)
        if duration > MAX_AUDIO_SECONDS:
            return error(413, f"Audio is {duration:.0f}s; the limit is {MAX_AUDIO_SECONDS:.0f}s.")
        fmt = "text"
        result = {}

        def work():
            try:
                result["raw"] = run_model(wav, fmt, prompt)
            except Exception as e:  # noqa: BLE001
                result["err"] = str(e)

        t0 = time.time()
        th = threading.Thread(target=work, daemon=True)
        th.start()
        await asyncio.to_thread(th.join, PROCESS_TIMEOUT)
        if th.is_alive():
            kill_runtime()
            await asyncio.to_thread(th.join, 5)
            threading.Thread(target=start_server, daemon=True).start()
            return error(504, f"Transcription exceeded {PROCESS_TIMEOUT:.0f}s; use shorter audio.", "timeout_error")
        if "err" in result:
            kill_runtime()
            threading.Thread(target=start_server, daemon=True).start()
            return error(500, "Runtime failed: " + result["err"], "server_error")
        raw = result["raw"]
        if raw.startswith("[ERROR]"):
            return error(400, raw)
        elapsed = time.time() - t0
        headers = {"X-Audio-Seconds": f"{duration:.2f}", "X-Processing-Seconds": f"{elapsed:.2f}"}
        if response_format == "text":
            return PlainTextResponse(raw, headers=headers)
        if response_format == "json":
            return JSONResponse({"text": raw}, headers=headers)
        return JSONResponse({"task": "transcribe", "language": language or None, "duration": duration,
                             "text": raw, "segments": []}, headers=headers)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        _lock.release()


def main():
    if not API_KEY:
        print("API_KEY is empty; refusing to start an unauthenticated server.", file=sys.stderr)
        sys.exit(1)
    for p in (ASR_BIN, LM_MODEL, VAE_MODEL):
        if not os.path.exists(p):
            print("missing:", p, file=sys.stderr)
            sys.exit(1)
    threading.Thread(target=start_server, daemon=True).start()
    # Railway's healthcheck dials the injected PORT over IPv4 and the private network is IPv6, and a "::"
    # listener is IPv6-only under asyncio, so one socket per family.
    port = int(os.environ.get("PORT") or 8080)
    socks = []
    for family, addr in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((addr, port))
            sock.set_inheritable(True)
            socks.append(sock)
        except OSError as exc:
            print(f"[app] cannot bind {addr}:{port}: {exc}", flush=True)
    if not socks:
        sys.exit(1)
    uvicorn.Server(uvicorn.Config(app, log_level=os.environ.get("LOG_LEVEL", "warning"))).run(sockets=socks)


if __name__ == "__main__":
    main()
