import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import time
from typing import Optional, List
from pathlib import Path
import torch
from transformers import AutoProcessor, DiaForConditionalGeneration
from utils import process_audio_prompt

# Set up structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Ensure directories exist
AUDIO_DIR = Path("audio_files")
AUDIO_DIR.mkdir(exist_ok=True)
UPLOADS_DIR = Path("upload_files")
UPLOADS_DIR.mkdir(exist_ok=True)

# Set computation device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using DEVICE: {DEVICE}")


class ModelManager:
    """Manages the loading, unloading and access to the Dia model and processor using Hugging Face Transformers."""

    def __init__(self):
        self.device = DEVICE
        self.dtype_map = {
            "cpu": "float32",
            "cuda": "float16",
        }
        self.model = None
        self.processor = None
        self.model_id = "nari-labs/Dia-1.6B-0626"

    def load_model(self):
        """Load the Dia model and processor with appropriate configuration using Hugging Face Transformers."""
        try:
            dtype = self.dtype_map.get(self.device, torch.float16)
            logger.info(f"Loading model and processor with {dtype} on {self.device}")
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = DiaForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=dtype, device_map=self.device
            )
            logger.info("Model and processor loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model or processor: {e}")
            raise

    def unload_model(self):
        """Cleanup method to properly unload the model."""
        try:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error unloading model: {e}")

    def get_model(self):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model

    def get_processor(self):
        if self.processor is None:
            raise RuntimeError("Processor not loaded. Call load_model() first.")
        return self.processor


model_manager = ModelManager()


# Request models
class AudioPrompt(BaseModel):
    sample_rate: int
    audio_data: List[float]


class GenerateRequest(BaseModel):
    text_input: str
    audio_prompt_input: Optional[AudioPrompt] = None
    max_new_tokens: int = 1024
    cfg_scale: float = 3.0
    temperature: float = 1.3
    top_p: float = 0.95
    cfg_filter_top_k: int = 35
    speed_factor: float = 0.94


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Handle model lifecycle during application startup and shutdown."""
    logger.info("Starting up application...")
    model_manager.load_model()
    yield
    logger.info("Shutting down application...")
    model_manager.unload_model()
    logger.info("Application shut down successfully")


app = FastAPI(
    title="Dia Text-to-Voice API",
    description="API for generating voice using Dia model",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "Backend is running"}


@app.post("/api/generate")
async def run_inference(request: GenerateRequest):
    """
    Runs Dia inference using the model and processor from model_manager and provided inputs.
    Uses temporary files for audio prompt compatibility with inference.generate.
    """
    if not request.text_input or request.text_input.isspace():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    output_filepath = AUDIO_DIR / f"{int(time.time())}.wav"

    try:
        prompt_path_for_generate = None
        if request.audio_prompt_input is not None:
            prompt_path_for_generate = process_audio_prompt(request.audio_prompt_input)

        model = model_manager.get_model()
        processor = model_manager.get_processor()

        start_time = time.time()
        processor_inputs = processor(
            text=[request.text_input],
            padding=True,
            return_tensors="pt"
        )
        processor_inputs = {k: v.to(model.device) for k, v in processor_inputs.items()}

        if prompt_path_for_generate is not None:
            processor_inputs["audio_prompt"] = prompt_path_for_generate

        with torch.inference_mode():
            logger.info(f"Starting generation with audio prompt: {prompt_path_for_generate}")
            outputs = model.generate(
                **processor_inputs,
                max_new_tokens=request.max_new_tokens,
                guidance_scale=request.cfg_scale,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.cfg_filter_top_k
            )
            logger.info(f"Generation completed. Output shape: {outputs.shape if hasattr(outputs, 'shape') else type(outputs)}")

        decoded = processor.batch_decode(outputs)
        processor.save_audio(decoded, str(output_filepath))
        logger.info(f"Audio saved to {output_filepath}")

        end_time = time.time()
        logger.info(f"Generation finished in {end_time - start_time:.2f} seconds.")

        return FileResponse(
            path=str(output_filepath),
            media_type="audio/wav",
            filename=output_filepath.name
        )

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
