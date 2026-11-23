# VibeVoice-ASR-BitNet speech-to-text API (OpenAI /v1/audio/transcriptions), Bearer API_KEY.
# Source: https://github.com/bon5co/vibeasr-railway
# CPU only (ggml, BitNet 1.58-bit weights), weights baked in (about 1.7 GB), stateless (no volume needed).
# upstream: microsoft/VibeASR.cpp built from the pinned commit below (newest commit on main, checked 2026-11-23; the repository has no releases or tags)
FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ARG VIBEASR_COMMIT=c4334009c88060f86cdbbd684b62662f710b6c20
RUN git clone --recursive https://github.com/microsoft/VibeASR.cpp.git /src \
    && git -C /src checkout ${VIBEASR_COMMIT} && git -C /src submodule update --init --recursive \
    && git -C /src/3rdparty/llama.cpp rev-parse HEAD
WORKDIR /src
# Fixed x86-64-v3 target (AVX2, FMA, F16C), never -march=native: the Actions runner that builds the image is not
# the host Railway schedules it on, and an AVX-512 build dies with SIGILL on a CPU without it.
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
        -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF \
        -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
    && cmake --build build --target asr_stream_server -j"$(nproc)" \
    && ls -l build/bin && ldd build/bin/asr_stream_server

# Weights, verified by sha256 so a changed or replaced file fails the build instead of shipping. A download at boot
# would sit inside Railway's healthcheck window. Both files are from microsoft/VibeVoice-ASR-BitNet, whose Hugging
# Face card front matter reads `license: mit`.
FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df AS weights
ARG HF=https://huggingface.co/microsoft/VibeVoice-ASR-BitNet/resolve/66e78021ab8f5f06133d1ab421ba4d348bda97c9
ADD --checksum=sha256:fbe273d8dc2f2433bb25f849e19d77ea65aaa2188d12c20cee987ab6f321e002 ${HF}/vibeasr-lm-i2_s-embed-q6_k.gguf /models/vibeasr-lm-i2_s-embed-q6_k.gguf
ADD --checksum=sha256:4941c82608c253ec066b5cc74d3dd11a5c8fef96cccbc5b87359ef0fe4338df6 ${HF}/vibeasr-vae-encoder-i8_s.gguf /models/vibeasr-vae-encoder-i8_s.gguf

FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin vibeasr
COPY --from=build /src/build/bin/asr_stream_server /usr/local/bin/asr_stream_server
COPY --from=weights /models /models
RUN chmod -R a+rX /models
WORKDIR /app
COPY app.py entrypoint.sh NOTICE LICENSE /app/
RUN chmod +x /app/entrypoint.sh
ENV MODEL_DIR=/models ASR_BIN=/usr/local/bin/asr_stream_server
USER 1000
EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "/app/app.py"]
