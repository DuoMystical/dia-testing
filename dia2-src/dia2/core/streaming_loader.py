"""
Streaming safetensors loader for direct GPU loading during download.

This module provides utilities to load model weights directly to GPU memory
as they stream from the network, rather than downloading to disk first.
"""

from __future__ import annotations

import json
import struct
import sys
from typing import Any, Dict, Optional, Iterator, Tuple
from io import BytesIO

import torch
from huggingface_hub import hf_hub_download, HfFileSystem
from huggingface_hub.utils import build_hf_headers


def parse_safetensors_header(data: bytes) -> Tuple[int, Dict[str, Any]]:
    """
    Parse the safetensors header from raw bytes.

    Args:
        data: Raw bytes starting from the beginning of the file

    Returns:
        Tuple of (header_size, header_dict)
        header_dict maps tensor names to {dtype, shape, data_offsets}
    """
    if len(data) < 8:
        raise ValueError("Need at least 8 bytes for header size")

    header_size = struct.unpack('<Q', data[:8])[0]

    if len(data) < 8 + header_size:
        raise ValueError(f"Need {8 + header_size} bytes for full header, got {len(data)}")

    header_json = data[8:8 + header_size].decode('utf-8')
    header = json.loads(header_json)

    return header_size, header


def get_tensor_info(header: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract tensor metadata from safetensors header."""
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        tensors[name] = {
            'dtype': info['dtype'],
            'shape': info['shape'],
            'start': info['data_offsets'][0],
            'end': info['data_offsets'][1],
        }
    return tensors


DTYPE_MAP = {
    'F32': torch.float32,
    'F16': torch.float16,
    'BF16': torch.bfloat16,
    'I64': torch.int64,
    'I32': torch.int32,
    'I16': torch.int16,
    'I8': torch.int8,
    'U8': torch.uint8,
    'BOOL': torch.bool,
}


def streaming_load_safetensors(
    repo_id: str,
    filename: str,
    model: torch.nn.Module,
    device: torch.device,
    *,
    revision: str = "main",
    token: Optional[str] = None,
    chunk_size: int = 8 * 1024 * 1024,  # 8MB chunks
) -> None:
    """
    Stream a safetensors file from HuggingFace Hub directly to GPU.

    This downloads and loads tensors progressively, allowing the model
    to start receiving weights as soon as they arrive over the network.

    Benefits:
    - GPU receives tensors while download is still in progress
    - Lower memory footprint (no need to hold entire file in RAM)
    - Faster time-to-ready for autoscaling cold starts

    Args:
        repo_id: HuggingFace repository ID (e.g., "nari-labs/Dia2-2B")
        filename: Name of the safetensors file in the repo
        model: PyTorch model to load weights into
        device: Target device for the tensors
        revision: Git revision (branch, tag, or commit)
        token: HuggingFace token for private repos
        chunk_size: Download chunk size in bytes
    """
    import requests
    import time
    from huggingface_hub import hf_hub_url

    # Build the URL for the file
    url = hf_hub_url(repo_id, filename, revision=revision)
    headers = build_hf_headers(token=token)

    print(f"[STREAMING] Starting download from {repo_id}/{filename}", file=sys.stderr)
    start_time = time.time()

    # Get file size first
    response = requests.head(url, headers=headers, allow_redirects=True)
    total_size = int(response.headers.get('content-length', 0))
    print(f"[STREAMING] Total file size: {total_size / 1024 / 1024:.1f} MB", file=sys.stderr)

    # Start streaming download
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()

    # First, read enough to get the header
    buffer = BytesIO()
    header_parsed = False
    header_size = 0
    tensor_info = {}
    data_start = 0

    state_dict = model.state_dict()
    loaded_tensors = set()
    bytes_downloaded = 0
    last_progress_time = start_time

    # Track which byte ranges we have
    data_buffer = BytesIO()
    data_buffer_start = 0  # Offset in file where data_buffer starts

    for chunk in response.iter_content(chunk_size=chunk_size):
        if not chunk:
            continue

        bytes_downloaded += len(chunk)

        if not header_parsed:
            buffer.write(chunk)
            buffer_data = buffer.getvalue()

            # Try to parse header
            if len(buffer_data) >= 8:
                try:
                    header_size, header = parse_safetensors_header(buffer_data)
                    tensor_info = get_tensor_info(header)
                    data_start = 8 + header_size
                    header_parsed = True

                    print(f"[STREAMING] Header parsed: {len(tensor_info)} tensors", file=sys.stderr)

                    # Initialize data buffer with any remaining bytes after header
                    remaining = buffer_data[data_start:]
                    data_buffer = BytesIO(remaining)
                    data_buffer_start = 0

                except ValueError:
                    # Need more data for header
                    continue
        else:
            # Append to data buffer
            current_pos = data_buffer.tell()
            data_buffer.seek(0, 2)  # Seek to end
            data_buffer.write(chunk)
            data_buffer.seek(current_pos)  # Restore position

        if header_parsed:
            # Try to load any complete tensors
            data_bytes = data_buffer.getvalue()
            data_len = len(data_bytes)

            for name, info in tensor_info.items():
                if name in loaded_tensors:
                    continue

                # Check if we have all data for this tensor
                tensor_start = info['start'] - data_buffer_start
                tensor_end = info['end'] - data_buffer_start

                if tensor_start >= 0 and tensor_end <= data_len:
                    # We have the complete tensor data
                    tensor_bytes = data_bytes[tensor_start:tensor_end]

                    # Convert to tensor
                    dtype = DTYPE_MAP.get(info['dtype'], torch.float32)
                    shape = info['shape']

                    # Load directly to device
                    tensor = torch.frombuffer(
                        bytearray(tensor_bytes),
                        dtype=dtype
                    ).reshape(shape).to(device)

                    # Copy into model's state dict
                    if name in state_dict:
                        state_dict[name].copy_(tensor)
                        loaded_tensors.add(name)

                    del tensor  # Free memory

            # Compact buffer by removing data we've fully processed
            min_needed = min(
                (info['start'] for name, info in tensor_info.items()
                 if name not in loaded_tensors),
                default=data_buffer_start + data_len
            )

            if min_needed > data_buffer_start:
                trim_amount = min_needed - data_buffer_start
                if trim_amount > 0 and trim_amount < data_len:
                    remaining = data_bytes[trim_amount:]
                    data_buffer = BytesIO(remaining)
                    data_buffer_start = min_needed

        # Progress update (throttled to every 0.5s)
        now = time.time()
        if now - last_progress_time >= 0.5:
            progress = bytes_downloaded / total_size * 100 if total_size > 0 else 0
            elapsed = now - start_time
            speed = bytes_downloaded / elapsed / 1024 / 1024 if elapsed > 0 else 0
            print(f"\r[STREAMING] {progress:.1f}% | {speed:.1f} MB/s | {len(loaded_tensors)}/{len(tensor_info)} tensors loaded",
                  end='', file=sys.stderr)
            last_progress_time = now

    elapsed = time.time() - start_time
    speed = bytes_downloaded / elapsed / 1024 / 1024 if elapsed > 0 else 0
    print(f"\n[STREAMING] Download complete in {elapsed:.1f}s ({speed:.1f} MB/s). Loaded {len(loaded_tensors)}/{len(tensor_info)} tensors",
          file=sys.stderr)

    # Verify all tensors loaded
    missing = set(tensor_info.keys()) - loaded_tensors
    if missing:
        print(f"[STREAMING] Warning: {len(missing)} tensors not in model state dict (expected for some architectures)", file=sys.stderr)
