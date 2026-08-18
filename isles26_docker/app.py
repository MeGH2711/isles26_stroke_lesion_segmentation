"""
Algorithm inference server for Grand Challenge (ISLES 2026).

Grand Challenge runs your algorithm as an HTTP server with two endpoints:
  GET  /health  — Polled repeatedly after the container starts.
  POST /invoke  — Called per case to process inputs.
"""

import threading
import time
from contextlib import asynccontextmanager

import inference
from inference import _log
import uvicorn
from fastapi import FastAPI, status
from uvicorn.config import LOGGING_CONFIG

MODEL = None
MODEL_LOCK = threading.Lock()
MODEL_LOADED_EVENT = threading.Event()
MODEL_ERROR = None


def _load_model_worker():
    """Load model in background thread so the HTTP server binds immediately."""
    global MODEL, MODEL_ERROR
    _log("Background worker: Starting model loading ...")
    t0 = time.time()
    try:
        loaded = inference.init_model()
        with MODEL_LOCK:
            MODEL = loaded
        _log(f"Background worker: Model loaded successfully in {time.time() - t0:.2f}s.")
    except Exception as e:
        MODEL_ERROR = e
        _log(f"Background worker: Error loading model: {e}", level="ERROR")
    finally:
        MODEL_LOADED_EVENT.set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start background loading thread immediately upon startup.
    Yields immediately so port 4743 is bound and responsive to /health.
    """
    thread = threading.Thread(target=_load_model_worker, daemon=True)
    thread.start()
    yield
    _log("Server shutting down ...")


app = FastAPI(lifespan=lifespan)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    """
    Responds immediately with HTTP 200 OK so Grand Challenge health check passes.
    """
    return {"status": "healthy"}


@app.post("/invoke", status_code=status.HTTP_201_CREATED)
async def invoke():
    """
    Called per test case. Ensures model loading has completed, then runs inference.
    """
    _log("POST /invoke received — ensuring model weights are ready ...")
    MODEL_LOADED_EVENT.wait()
    if MODEL_ERROR is not None:
        _log(f"Model failed to load: {MODEL_ERROR}", level="ERROR")
        raise RuntimeError(f"Model failed to load: {MODEL_ERROR}")

    _log("Model ready — starting inference pipeline ...")
    inference.run(MODEL)
    _log("POST /invoke successfully finished.")
    return {"status": "Inference complete"}


if __name__ == "__main__":
    LOGGING_CONFIG["loggers"]["uvicorn.access"]["level"] = "WARNING"
    uvicorn.run(app, host="0.0.0.0", port=4743)

