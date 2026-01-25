# Dia2 TTS API with Web Terminal for development
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies + dev tools
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    curl \
    wget \
    ttyd \
    libsndfile1 \
    vim \
    nano \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY main.py utils.py ./

# Create virtual environment and install dependencies
RUN uv venv /app/.venv --python python3.12
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Install dependencies with uv
RUN uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    fastapi[standard]>=0.128.0 \
    numpy>=2.1.0 \
    pydantic>=2.12.5 \
    soundfile>=0.12.1 \
    torch>=2.8.0 \
    transformers>=4.55.3 \
    safetensors>=0.5.3 \
    huggingface-hub>=0.24.7 \
    sphn>=0.2.0 \
    whisper-timestamped>=1.14.2

# Install dia2 from git (editable for development)
RUN git clone --depth 1 https://github.com/nari-labs/dia2.git /app/dia2-src && \
    cd /app/dia2-src && \
    uv pip install -e .

# Create startup script
COPY <<'STARTUP' /app/start.sh
#!/bin/bash
set -e

# Default credentials if not set
TERMINAL_USER=${TERMINAL_USER:-admin}
TERMINAL_PASS=${TERMINAL_PASS:-dia2dev}

echo "Starting Dia2 API on port 8000..."
echo "Starting Web Terminal on port 7681 (user: $TERMINAL_USER)"

# Start ttyd in background
ttyd -p 7681 -W -c "$TERMINAL_USER:$TERMINAL_PASS" bash &

# Start uvicorn in foreground
exec /app/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
STARTUP

RUN chmod +x /app/start.sh

# Expose ports: 8000 for API, 7681 for terminal
EXPOSE 8000 7681

# Environment variables for terminal auth (override in Koyeb)
ENV TERMINAL_USER=admin
ENV TERMINAL_PASS=dia2dev

CMD ["/app/start.sh"]
