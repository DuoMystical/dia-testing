from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import warnings

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..config import DiaConfig, load_config
from ..core.model import Dia2Model
from ..core.precision import Precision, resolve_precision
from ..audio import MimiCodec, DEFAULT_MIMI_MODEL_ID
from .state_machine import StateMachine, TokenIds


def load_file_into_model(
    model: torch.nn.Module,
    filename: str,
    device: str = "cpu",
) -> None:
    """
    Lazily loads a safetensors file sequentially into a model, avoiding holding
    the entire state dict in memory at once.

    Reduces peak memory compared to loading the full state dict and then calling
    `model.load_state_dict()`.

    Args:
        model (`nn.Module`):
            The model to load weights into
        filename (`str`):
            The name of the file which contains the tensors
        device (`str`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.

    Example:
    ```python
    model = MyModel()
    load_file_into_model(model, "./my_folder/bert.safetensors", device="cuda")
    ```
    """
    state_dict = model.state_dict() # This is a shallow copy.
    with safe_open(filename, framework="pt", device=device) as f:
        for key in f.keys():
            if key in state_dict:
                state_dict[key].copy_(f.get_tensor(key))


@dataclass
class RuntimeContext:
    config: DiaConfig
    model: Dia2Model
    precision: Precision
    tokenizer: PreTrainedTokenizerBase
    mimi: MimiCodec
    device: torch.device
    machine: StateMachine
    transformer_step: callable
    depformer_step: callable
    constants: TokenIds
    audio_delays: list[int]
    audio_delay_tensor: torch.Tensor
    frame_rate: float


def build_runtime(
    *,
    config_path: str | Path,
    weights_path: str | Path,
    tokenizer_id: Optional[str],
    repo_id: Optional[str],
    mimi_id: Optional[str],
    device: str,
    dtype_pref: str,
    weights_repo_id: Optional[str] = None,
    weights_filename: Optional[str] = None,
) -> tuple[RuntimeContext, str, str]:
    import time as time_module
    import sys

    device_obj = torch.device(device)
    if device_obj.type == "cuda":
        cuda_matmul = torch.backends.cuda.matmul
        cudnn_conv = torch.backends.cudnn.conv
        if hasattr(cuda_matmul, "fp32_precision"):
            cuda_matmul.fp32_precision = "tf32"
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Please use the new API settings",
                )
                torch.backends.cuda.matmul.allow_tf32 = True
        else:  # pragma: no cover - compatibility with older PyTorch
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(cudnn_conv, "fp32_precision"):
            cudnn_conv.fp32_precision = "tf32"
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Please use the new API settings",
                )
                torch.backends.cudnn.allow_tf32 = True
        else:  # pragma: no cover
            torch.backends.cudnn.allow_tf32 = True

    precision = resolve_precision(dtype_pref, device_obj)
    config = load_config(config_path)

    print(f"[TIMING] Creating Dia2Model...", file=sys.stderr)
    t0 = time_module.time()
    model = Dia2Model(config, precision, device=device_obj)
    print(f"[TIMING] Dia2Model init: {time_module.time() - t0:.2f}s", file=sys.stderr)

    # Load weights - use streaming if not cached
    if weights_repo_id and weights_filename:
        # Stream weights directly to GPU during download
        print(f"[TIMING] Streaming weights from {weights_repo_id}/{weights_filename}...", file=sys.stderr)
        t0 = time_module.time()
        from ..core.streaming_loader import streaming_load_safetensors
        streaming_load_safetensors(
            weights_repo_id,
            weights_filename,
            model,
            device_obj,
        )
        print(f"[TIMING] Streaming weight load: {time_module.time() - t0:.2f}s", file=sys.stderr)
    else:
        # Load from cached local file
        print(f"[TIMING] Loading weights from {weights_path}...", file=sys.stderr)
        t0 = time_module.time()
        load_file_into_model(model, str(weights_path), device=device)
        print(f"[TIMING] Weight loading: {time_module.time() - t0:.2f}s", file=sys.stderr)

    tokenizer_ref = tokenizer_id or config.assets.tokenizer or repo_id
    if tokenizer_ref is None:
        raise ValueError("Tokenizer id is missing. Provide --tokenizer or add assets.tokenizer to the config.")

    print(f"[TIMING] Loading tokenizer...", file=sys.stderr)
    t0 = time_module.time()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_ref,
        use_fast=False,
        trust_remote_code=True,
    )
    print(f"[TIMING] Tokenizer: {time_module.time() - t0:.2f}s", file=sys.stderr)

    mimi_ref = mimi_id or config.assets.mimi or DEFAULT_MIMI_MODEL_ID
    print(f"[TIMING] Loading Mimi codec...", file=sys.stderr)
    t0 = time_module.time()
    mimi = MimiCodec.from_pretrained(mimi_ref, device=device_obj)
    print(f"[TIMING] Mimi codec: {time_module.time() - t0:.2f}s", file=sys.stderr)

    data_cfg = config.data
    constants = TokenIds(
        card=data_cfg.text_vocab_size,
        new_word=data_cfg.text_new_word_token_id,
        pad=data_cfg.text_pad_token_id,
        bos=getattr(tokenizer, "bos_token_id", 1) or 1,
        zero=data_cfg.text_zero_token_id,
        spk1=tokenizer.convert_tokens_to_ids("[S1]") if "[S1]" in tokenizer.get_vocab() else data_cfg.text_new_word_token_id,
        spk2=tokenizer.convert_tokens_to_ids("[S2]") if "[S2]" in tokenizer.get_vocab() else data_cfg.text_new_word_token_id,
        audio_pad=data_cfg.audio_pad_token_id,
        audio_bos=data_cfg.audio_bos_token_id,
    )
    machine = StateMachine(
        token_ids=constants,
        second_stream_ahead=data_cfg.second_stream_ahead,
        max_padding=6,
        initial_padding=0,
    )
    audio_delays = list(data_cfg.delay_pattern)
    audio_delay_tensor = torch.tensor(audio_delays, device=device_obj, dtype=torch.long) if audio_delays else torch.empty(0, dtype=torch.long, device=device_obj)
    frame_rate = getattr(mimi, "frame_rate", 75.0)

    runtime = RuntimeContext(
        config=config,
        precision=precision,
        model=model,
        tokenizer=tokenizer,
        mimi=mimi,
        device=device_obj,
        machine=machine,
        constants=constants,
        audio_delays=audio_delays,
        audio_delay_tensor=audio_delay_tensor,
        frame_rate=frame_rate,
        transformer_step=model.transformer.forward_step,
        depformer_step=model.depformer.forward_step,
    )
    return runtime, tokenizer_ref, mimi_ref


__all__ = [
    "RuntimeContext",
    "build_runtime",
]
