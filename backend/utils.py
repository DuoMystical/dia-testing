import logging
import numpy as np
import tempfile
import soundfile as sf
from fastapi import HTTPException
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def is_audio_empty_or_silent(audio_data: np.ndarray) -> bool:
    """Check if audio data is empty, None, or silent."""
    return audio_data is None or audio_data.size == 0 or audio_data.max() == 0


def normalize_audio_dtype(audio_data: np.ndarray) -> np.ndarray:
    """Convert audio data to float32 format."""
    if np.issubdtype(audio_data.dtype, np.integer):
        max_val = np.iinfo(audio_data.dtype).max
        return audio_data.astype(np.float32) / max_val
    elif not np.issubdtype(audio_data.dtype, np.floating):
        logger.warning(f"Unsupported audio prompt dtype {audio_data.dtype}, attempting conversion.")
        try:
            return audio_data.astype(np.float32)
        except Exception as conv_e:
            raise HTTPException(status_code=400, detail=f"Failed to convert audio prompt to float32: {conv_e}")
    return audio_data


def convert_to_mono(audio_data: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono."""
    if audio_data.ndim > 1:
        if audio_data.shape[0] == 2:
            audio_data = np.mean(audio_data, axis=0)
        elif audio_data.shape[1] == 2:
            audio_data = np.mean(audio_data, axis=1)
        else:
            logger.warning(f"Audio prompt has unexpected shape {audio_data.shape}, taking first channel/axis.")
            audio_data = audio_data[0] if audio_data.shape[0] < audio_data.shape[1] else audio_data[:, 0]
    audio_data = np.ascontiguousarray(audio_data)
    return audio_data


def save_audio_to_temp_file(audio_data: np.ndarray, sample_rate: int) -> str:
    """Save audio data to a temporary WAV file and return the file path."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f_audio:
        temp_path = f_audio.name
    try:
        sf.write(temp_path, audio_data, sample_rate, subtype="FLOAT")
        logger.info(f"Created temporary audio prompt file: {temp_path} (sr: {sample_rate}, shape: {audio_data.shape}, max: {audio_data.max():.3f}, min: {audio_data.min():.3f})")
        return temp_path
    except Exception as write_e:
        logger.error(f"Error writing temporary audio file: {write_e}")
        raise HTTPException(status_code=400, detail=f"Failed to save audio prompt: {write_e}")


def process_audio_prompt(audio_prompt) -> Optional[str]:
    """
    Process the audio prompt input and return the path to the temporary audio file.
    Returns None if the audio is empty or silent.
    """
    audio_data = np.array(audio_prompt.audio_data, dtype=np.float32)
    sample_rate = audio_prompt.sample_rate

    if is_audio_empty_or_silent(audio_data):
        logger.warning("Audio prompt seems empty or silent, ignoring prompt.")
        return None

    logger.info(f"Processing audio prompt: shape={audio_data.shape}, sample_rate={sample_rate}, dtype={audio_data.dtype}")

    audio_data = normalize_audio_dtype(audio_data)
    audio_data = convert_to_mono(audio_data)

    return save_audio_to_temp_file(audio_data, sample_rate)
