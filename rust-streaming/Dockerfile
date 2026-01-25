# Dia2 Streaming WebSocket Server
# Multi-stage build: Rust binary + Python environment
# NOTE: Build context is repo root for Koyeb

# Stage 1: Build Rust binary
FROM rust:1.75-slim AS rust-builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy Rust source (paths relative to repo root)
COPY rust-streaming/Cargo.toml rust-streaming/Cargo.lock ./
COPY rust-streaming/src ./src

# Build release binary
RUN cargo build --release

# Stage 2: Runtime with CUDA + Python + Rust binary
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    curl \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Create virtual environment
RUN uv venv /app/.venv --python python3.12
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Install Python dependencies
RUN uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch>=2.8.0 \
    transformers>=4.55.3 \
    safetensors>=0.5.3 \
    huggingface-hub>=0.24.7 \
    sphn>=0.2.0 \
    numpy>=2.1.0 \
    soundfile>=0.12.1

# Copy modified Dia2 source and install (paths relative to repo root)
COPY dia2-src /app/dia2-src
RUN cd /app/dia2-src && uv pip install -e .

# Copy Rust binary from builder
COPY --from=rust-builder /app/target/release/dia2-streaming-server /app/

# Copy TTS bridge and frontend (paths relative to repo root)
COPY rust-streaming/tts_bridge.py /app/tts_bridge.py
COPY streaming-frontend /app/frontend

# Environment variables
ENV TTS_BRIDGE_PATH=/app/tts_bridge.py
ENV PORT=8080
ENV RUST_LOG=dia2_streaming_server=info

EXPOSE 8080

CMD ["/app/dia2-streaming-server"]
