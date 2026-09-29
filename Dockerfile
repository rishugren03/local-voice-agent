# syntax=docker/dockerfile:1
#
# The voice agent, containerised. Everything CPU-only and local: no GPU, no
# cloud API, nothing reaches outside the host except the two model downloads
# the agent makes at startup (both overridable — see models/ below).
#
# Three build stages, because the two heavy dependencies have very different
# shapes: whisper.cpp is a C++ program that needs a full toolchain to compile
# but is needed at runtime only as one static-ish binary, and torch is a
# multi-hundred-megabyte wheel that silero-vad needs. Compiling in throwaway
# stages keeps gcc, cmake and the CMake object tree out of the final image.

# ---------------------------------------------------------------------------
# Stage 1: whisper.cpp -> a single CLI binary
# ---------------------------------------------------------------------------
# Pinned to the exact revision the host runs, so a container turn sounds like a
# host turn. A float here would silently change transcription quality, which is
# the kind of difference you only notice in an eval.
FROM debian:bookworm-slim AS whisper-build

ARG WHISPER_CPP_REV=080bbbe85230f624f0b52127f1ae1218247989f9

# make is not implied by cmake: without it CMake has no generator to fall back
# on and stops with "unable to find a build program corresponding to Unix
# Makefiles".
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        g++ \
        git \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone https://github.com/ggerganov/whisper.cpp.git . \
    && git checkout ${WHISPER_CPP_REV}

# CPU only, and no BLAS: this is a single-stream transcription service, so the
# OpenMP/BLAS paths would add container size and thread contention for no
# measurable win. -DGGML_NATIVE=OFF keeps the build from baking in the build
# host's CPU features, which would crash on a different client machine.
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DGGML_BLAS=OFF \
        -DWHISPER_BUILD_TESTS=OFF \
        -DWHISPER_BUILD_EXAMPLES=ON \
    && cmake --build build --config Release -j "$(nproc)" --target whisper-cli

# ---------------------------------------------------------------------------
# Stage 2: torch CPU wheels
# ---------------------------------------------------------------------------
# The agent imports torch for exactly one call — torch.from_numpy, to hand a
# float array to the Silero VAD — but pip resolves that to the default CUDA
# build, dragging in several gigabytes of nvidia-* wheels for a CPU-only service.
# Building the wheel here against the CPU index keeps the final image to a few
# hundred megabytes and leaves the runtime stage free of a package index
# configuration.
FROM python:3.12-slim AS torch-build

RUN pip install --no-cache-dir torch==2.13.0 torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cpu

# ---------------------------------------------------------------------------
# Stage 3: runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# libgomp1 is the OpenMP runtime that torch and whisper.cpp's ggml link against;
# without it both die at import with a confusing symbol error. The rest are the
# shared libraries Silero's ONNX voice-detector and Piper's onnxruntime need.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # The default 15s socket timeout aborts a multi-hundred-megabyte download on
    # a slow link, and the retry is wasted work in a build. Generous timeouts
    # plus more retries cost nothing when the network is fine.
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    # The code resolves WHISPER_BIN as a bare command name, and a stale system
    # binary of the same name would otherwise win.
    PATH=/opt/whisper/lib:$PATH \
    # Paired with the PATH above: the .so files live beside the executable.
    LD_LIBRARY_PATH=/opt/whisper/lib

# The binary alone is not enough: whisper-cli links dynamically against
# libwhisper and the ggml CPU libraries, so copying only the executable yields an
# image where it fails at startup with "libwhisper.so.1: cannot open shared
# object file". The whole build/bin directory is copied instead — binary and
# libwhisper/ggml .so files together, which is what whisper.cpp's own install
# layout expects.
COPY --from=whisper-build /src/build/bin/ /opt/whisper/lib/
COPY --from=torch-build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Piper renders the agent's speech. The voice itself is a .onnx model and is NOT
# baked in — see README, "Running in Docker", for where it comes from. A 60MB
# model in the image would mean a rebuild for every voice change.
#
# The BuildKit cache mount holds pip's download cache for both this and the
# requirements install below. These two layers fetch ~200MB of wheels between
# them, and one reset connection mid-download would otherwise discard every byte
# already fetched and restart the layer. Nothing is committed into the image, so
# the final size is unchanged — a retried build just resumes.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 --timeout 300 piper-tts==1.6.0

# Dependencies first, on their own layer, so editing the agent source does not
# invalidate the (slow) torch install. The runtime needs only a slice of
# requirements.txt: the CUDA nvidia-*, torchaudio and torchcodec entries are
# build-host artifacts of a GPU install and are skipped explicitly, because
# copying torch in from the CPU stage above is what actually satisfies them.
# (Piper's own dependencies land here too, since it is installed above.)
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -viE '^(torch|torchaudio|torchcodec|nvidia-|cuda-|triton)' \
        /tmp/requirements.txt > /tmp/requirements-runtime.txt \
    && pip install --retries 10 --timeout 300 \
        -r /tmp/requirements-runtime.txt

COPY agent_platform/ /app/agent_platform/
COPY .env.example /app/.env.example
COPY mcp_server.py transcribe_test.py config_cli.py score_eval.py \
     build_dashboard_data.py analyze_concurrency.py \
     test_scoring.py test_session_isolation.py test_degradation.py \
     dashboard.html /app/
# The dashboard references the logo, so it has to come along or the served page
# shows a broken image in the container.
COPY assets/ /app/assets/

# Models live on a volume, not in the image. The directory exists and is owned
# by the unprivileged user so a host bind-mount or a named volume both work.
# /scratch is where the agent writes the per-turn .wav files (STT input, TTS
# output). It is a directory rather than the CWD because /app is not writable by
# the unprivileged user, and a chunk that cannot be written kills the turn after
# the caller has already spoken.
RUN mkdir -p /models/whisper /models/piper /data /scratch \
    && useradd --create-home --uid 10001 agent \
    && chown -R agent:agent /app /models /data /scratch

USER agent

# WHISPER_BIN/PIPER_MODEL are overridden by docker-compose, which is where the
# mounted model paths are defined. These are the in-container defaults so the
# image is also runnable on its own.
ENV WHISPER_BIN=/opt/whisper/lib/whisper-cli \
    WHISPER_MODEL=/models/whisper/ggml-base.en.bin \
    PIPER_MODEL=/models/piper/en_US-lessac-medium.onnx \
    TRACE_DB=/data/call_trace.db \
    AGENT_SCRATCH_DIR=/scratch

EXPOSE 7880

# Readiness is the LiveKit URL being reachable, not the process existing: the
# agent exits immediately on a bad URL, and a healthcheck on PID alone would
# report healthy for a container that is about to die.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.parse,socket,sys; u=urllib.parse.urlparse(os.environ['LIVEKIT_URL']); s=socket.create_connection((u.hostname, u.port or 7880), 3); s.close()" || exit 1

CMD ["python", "transcribe_test.py"]
