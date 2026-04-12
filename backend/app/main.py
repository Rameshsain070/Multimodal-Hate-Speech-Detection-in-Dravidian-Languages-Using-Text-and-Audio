from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

from .model_service import MultimodalService


app = FastAPI(title="Dravidian Multimodal Hate Speech API", version="1.0.0")
service = MultimodalService()
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(10 * 1024 * 1024)))
PREDICT_TIMEOUT_SECONDS = int(os.getenv("PREDICT_TIMEOUT_SECONDS", "120"))
WARMUP_MODELS = os.getenv("WARMUP_MODELS", "true").strip().lower() in {"1", "true", "yes", "on"}
WARMUP_ALL_MODELS = os.getenv("WARMUP_ALL_MODELS", "false").strip().lower() in {"1", "true", "yes", "on"}
allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500")
allowed_origins = [origin.strip() for origin in allowed_origins_raw.split(",") if origin.strip()]
allow_origin_regex = os.getenv("ALLOWED_ORIGIN_REGEX", r"^https://[a-zA-Z0-9-]+\.github\.io$").strip() or None
allow_all_origins = "*" in allowed_origins
job_store: Dict[str, dict] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins else allowed_origins,
    allow_origin_regex=None if allow_all_origins else allow_origin_regex,
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_model_keys(raw_value: Optional[str]) -> Optional[list[str]]:
    if raw_value is None:
        return None
    items = [item.strip().lower() for item in raw_value.split(",") if item.strip()]
    return items if items else []


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _predict_with_limits(
    language: str,
    text: Optional[str],
    audio_bytes: Optional[bytes],
    text_models_raw: Optional[str],
    audio_models_raw: Optional[str],
) -> dict:
    text_model_keys = _parse_model_keys(text_models_raw)
    audio_model_keys = _parse_model_keys(audio_models_raw)
    return await asyncio.wait_for(
        run_in_threadpool(
            service.predict,
            language,
            text,
            audio_bytes,
            text_model_keys,
            audio_model_keys,
        ),
        timeout=PREDICT_TIMEOUT_SECONDS,
    )


@app.on_event("startup")
async def startup_event() -> None:
    if WARMUP_MODELS:
        await run_in_threadpool(service.warmup, WARMUP_ALL_MODELS)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "languages": service.supported_languages(),
        "timeout_seconds": PREDICT_TIMEOUT_SECONDS,
        "warmup_models": WARMUP_MODELS,
        "warmup_all_models": WARMUP_ALL_MODELS,
    }


@app.get("/models")
def models() -> dict:
    return {"languages": service.model_catalog()}


@app.post("/predict")
async def predict(
    language: str = Form(...),
    text: Optional[str] = Form(default=None),
    audio: Optional[UploadFile] = File(default=None),
    text_models: Optional[str] = Form(default=None),
    audio_models: Optional[str] = Form(default=None),
) -> dict:
    try:
        audio_bytes = await audio.read() if audio is not None else None
        if audio_bytes == b"":
            audio_bytes = None
        if audio_bytes is not None and len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file too large. Max allowed size is {MAX_AUDIO_BYTES} bytes.",
            )
        if not (text and text.strip()) and audio_bytes is None:
            raise HTTPException(status_code=400, detail="Provide text or audio input.")
        return await _predict_with_limits(
            language=language,
            text=text,
            audio_bytes=audio_bytes,
            text_models_raw=text_models,
            audio_models_raw=audio_models,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Prediction timed out after {PREDICT_TIMEOUT_SECONDS} seconds.",
        ) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict/jobs")
async def predict_job(
    language: str = Form(...),
    text: Optional[str] = Form(default=None),
    audio: Optional[UploadFile] = File(default=None),
    text_models: Optional[str] = Form(default=None),
    audio_models: Optional[str] = Form(default=None),
) -> JSONResponse:
    audio_bytes = await audio.read() if audio is not None else None
    if audio_bytes == b"":
        audio_bytes = None
    if audio_bytes is not None and len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large. Max allowed size is {MAX_AUDIO_BYTES} bytes.",
        )
    if not (text and text.strip()) and audio_bytes is None:
        raise HTTPException(status_code=400, detail="Provide text or audio input.")

    job_id = str(uuid.uuid4())
    job_store[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "result": None,
        "error": None,
    }

    async def _run_job() -> None:
        job_store[job_id]["status"] = "running"
        job_store[job_id]["updated_at"] = _now_iso()
        try:
            result = await _predict_with_limits(
                language=language,
                text=text,
                audio_bytes=audio_bytes,
                text_models_raw=text_models,
                audio_models_raw=audio_models,
            )
            job_store[job_id]["status"] = "completed"
            job_store[job_id]["result"] = result
        except Exception as exc:  # pragma: no cover
            job_store[job_id]["status"] = "failed"
            job_store[job_id]["error"] = str(exc)
        finally:
            job_store[job_id]["updated_at"] = _now_iso()

    asyncio.create_task(_run_job())

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "queued",
            "status_url": f"/predict/jobs/{job_id}",
        },
    )


@app.get("/predict/jobs/{job_id}")
def predict_job_status(job_id: str) -> dict:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job
