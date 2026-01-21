import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import time
from typing import Optional, List, Literal
from pathlib import Path
import torch

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

# Set computation device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16" if DEVICE == "cuda" else "float32"
logger.info(f"Using DEVICE: {DEVICE}, DTYPE: {DTYPE}")


class Dia2ModelManager:
    """Manages loading and inference for Dia2 models (1B and 2B variants)."""

    def __init__(self):
        self.device = DEVICE
        self.dtype = DTYPE
        self.models = {}  # Store both models: {"1b": model, "2b": model}
        self.model_repos = {
            "1b": "nari-labs/Dia2-1B",
            "2b": "nari-labs/Dia2-2B",
        }

    def load_models(self):
        """Load both Dia2-1B and Dia2-2B models at startup."""
        from dia2 import Dia2

        for size, repo in self.model_repos.items():
            try:
                logger.info(f"Loading Dia2-{size.upper()} from {repo}...")
                model = Dia2.from_repo(
                    repo,
                    device=self.device,
                    dtype=self.dtype,
                )
                self.models[size] = model
                logger.info(f"Dia2-{size.upper()} loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Dia2-{size.upper()}: {e}")
                raise

    def unload_models(self):
        """Cleanup method to properly unload models."""
        try:
            for size in list(self.models.keys()):
                del self.models[size]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("All models unloaded successfully")
        except Exception as e:
            logger.error(f"Error unloading models: {e}")

    def get_model(self, size: str):
        """Get a specific model by size (1b or 2b)."""
        size = size.lower()
        if size not in self.models:
            raise RuntimeError(f"Model Dia2-{size.upper()} not loaded.")
        return self.models[size]

    def generate(
        self,
        text: str,
        model_size: str = "2b",
        # Audio sampling config
        audio_temperature: float = 0.8,
        audio_top_k: int = 50,
        # Text sampling config
        text_temperature: float = 0.6,
        text_top_k: int = 50,
        # Generation config
        cfg_scale: float = 2.0,
        cfg_filter_k: int = 50,
        use_cuda_graph: bool = True,
        use_torch_compile: bool = False,
        output_path: str = None,
    ):
        """Generate audio using the specified model with all Dia2 options."""
        from dia2 import GenerationConfig, SamplingConfig

        model = self.get_model(model_size)

        # Build the full configuration
        config = GenerationConfig(
            text=SamplingConfig(temperature=text_temperature, top_k=text_top_k),
            audio=SamplingConfig(temperature=audio_temperature, top_k=audio_top_k),
            cfg_scale=cfg_scale,
            cfg_filter_k=cfg_filter_k,
            use_cuda_graph=use_cuda_graph,
            use_torch_compile=use_torch_compile,
        )

        result = model.generate(
            text,
            config=config,
            output_wav=output_path,
            verbose=True,
        )

        return result


model_manager = Dia2ModelManager()


# Request models
class GenerateRequest(BaseModel):
    text_input: str
    model: Literal["1b", "2b"] = "2b"  # Default to 2B model

    # Audio sampling (affects voice quality/variation)
    audio_temperature: float = 0.8  # Higher = more variation in voice
    audio_top_k: int = 50  # Top-k sampling for audio tokens

    # Text sampling (affects pronunciation/pacing)
    text_temperature: float = 0.6  # Higher = more variation in text interpretation
    text_top_k: int = 50  # Top-k sampling for text tokens

    # Generation config
    cfg_scale: float = 2.0  # Classifier-free guidance scale (higher = more faithful to text)
    cfg_filter_k: int = 50  # CFG filter top-k
    use_cuda_graph: bool = True  # CUDA graph optimization
    use_torch_compile: bool = False  # torch.compile optimization (slower first run)


class ModelsResponse(BaseModel):
    available_models: List[str]
    default_model: str


class ModelInfo(BaseModel):
    name: str
    parameters: str
    description: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Handle model lifecycle during application startup and shutdown."""
    logger.info("Starting up application...")
    model_manager.load_models()
    yield
    logger.info("Shutting down application...")
    model_manager.unload_models()
    logger.info("Application shut down successfully")


app = FastAPI(
    title="Dia2 Text-to-Voice API",
    description="API for generating voice using Dia2 models (1B and 2B variants). Max generation length: 2 minutes.",
    version="2.0.0",
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
    loaded_models = list(model_manager.models.keys())
    return {
        "status": "ok",
        "message": "Backend is running",
        "device": DEVICE,
        "dtype": DTYPE,
        "loaded_models": [f"dia2-{m}" for m in loaded_models],
        "max_generation_length": "2 minutes (1500 context steps)",
    }


@app.get("/api/models", response_model=ModelsResponse)
async def get_available_models():
    """Get list of available models."""
    return ModelsResponse(
        available_models=["1b", "2b"],
        default_model="2b",
    )


@app.get("/api/models/info")
async def get_models_info():
    """Get detailed info about available models."""
    return [
        ModelInfo(
            name="dia2-1b",
            parameters="1 billion",
            description="Faster, lighter model. Good for quick iterations.",
        ),
        ModelInfo(
            name="dia2-2b",
            parameters="2 billion",
            description="Better quality model. Recommended for final output.",
        ),
    ]


@app.post("/api/generate")
async def run_inference(request: GenerateRequest):
    """
    Generate audio using Dia2 model.

    - model: "1b" for Dia2-1B (faster, lighter) or "2b" for Dia2-2B (better quality)
    - Max generation length: 2 minutes (1500 context steps)
    - Use [S1] and [S2] tags for different speakers
    """
    if not request.text_input or request.text_input.isspace():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    output_filepath = AUDIO_DIR / f"{int(time.time())}_{request.model}.wav"

    try:
        start_time = time.time()
        logger.info(f"Starting generation with Dia2-{request.model.upper()}")
        logger.info(f"Config: cfg_scale={request.cfg_scale}, audio_temp={request.audio_temperature}, text_temp={request.text_temperature}")
        logger.info(f"Text: {request.text_input[:100]}...")

        model_manager.generate(
            text=request.text_input,
            model_size=request.model,
            audio_temperature=request.audio_temperature,
            audio_top_k=request.audio_top_k,
            text_temperature=request.text_temperature,
            text_top_k=request.text_top_k,
            cfg_scale=request.cfg_scale,
            cfg_filter_k=request.cfg_filter_k,
            use_cuda_graph=request.use_cuda_graph,
            use_torch_compile=request.use_torch_compile,
            output_path=str(output_filepath),
        )

        end_time = time.time()
        generation_time = end_time - start_time
        logger.info(f"Generation finished in {generation_time:.2f} seconds using Dia2-{request.model.upper()}")

        return FileResponse(
            path=str(output_filepath),
            media_type="audio/wav",
            filename=output_filepath.name,
            headers={
                "X-Generation-Time": str(generation_time),
                "X-Model-Used": f"dia2-{request.model}",
            }
        )

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
