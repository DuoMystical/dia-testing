#!/bin/bash
set -e

echo "Building Dia2 Streaming Server..."

# Build in release mode
cargo build --release

echo "Build complete!"
echo "Binary at: target/release/dia2-streaming-server"

# Create symlink for convenience
ln -sf target/release/dia2-streaming-server ./server

echo ""
echo "To run:"
echo "  ./server"
echo "  or"
echo "  TTS_BRIDGE_PATH=./tts_bridge.py PORT=8080 ./server"
