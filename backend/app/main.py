from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .model_service import MultimodalService


app = FastAPI(title="Dravidian Multimodal Hate Speech API", version="1.0.0")
service = MultimodalService()
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(10 * 1024 * 1024)))
allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500")
allowed_origins = [origin.strip() for origin in allowed_origins_raw.split(",") if origin.strip()]
allow_origin_regex = os.getenv("ALLOWED_ORIGIN_REGEX", r"^https://[a-zA-Z0-9-]+\.github\.io$").strip() or None
allow_all_origins = "*" in allowed_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins else allowed_origins,
    allow_origin_regex=None if allow_all_origins else allow_origin_regex,
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "languages": service.supported_languages()}


@app.post("/predict")
async def predict(
    language: str = Form(...),
    text: str = Form(...),
    audio: Optional[UploadFile] = File(default=None),
) -> dict:
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text input is required.")

    try:
        audio_bytes = await audio.read() if audio is not None else None
        if audio_bytes == b"":
            audio_bytes = None
        if audio_bytes is not None and len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file too large. Max allowed size is {MAX_AUDIO_BYTES} bytes.",
            )
        return service.predict(language=language, text=text, audio_bytes=audio_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
