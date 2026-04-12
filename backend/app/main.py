from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .model_service import MultimodalService


app = FastAPI(title="Dravidian Multimodal Hate Speech API", version="1.0.0")
service = MultimodalService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
    audio: UploadFile | None = File(default=None),
) -> dict:
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text input is required.")

    try:
        audio_bytes = await audio.read() if audio is not None else None
        if audio_bytes == b"":
            audio_bytes = None
        return service.predict(language=language, text=text, audio_bytes=audio_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
