from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import librosa
import numpy as np
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


LABELS = {0: "Non-Hate Speech", 1: "Hate Speech"}


@dataclass(frozen=True)
class LanguageConfig:
    text_model: str
    audio_model: str
    meta_model_file: str


LANGUAGE_CONFIGS: Dict[str, LanguageConfig] = {
    "tamil": LanguageConfig(
        text_model="ramesh070/xlm-r-hatespeechdetection-tamil",
        audio_model="ramesh070/wav2vec2-tamil-binary",
        meta_model_file="meta_lightgbm_tamil_from_eval_safe.joblib",
    ),
    "telugu": LanguageConfig(
        text_model="ramesh070/xlm-r-hatespeechdetection-telugu",
        audio_model="ramesh070/wav2vec2-telugu-binary",
        meta_model_file="meta_lightgbm_telugu_from_eval_safe.joblib",
    ),
    "malayalam": LanguageConfig(
        text_model="ramesh070/xlm-r-hatespeechdetection-malayalam",
        audio_model="ramesh070/wav2vec2-malayalam-binary",
        meta_model_file="meta_lightgbm_malayalam_from_eval_safe.joblib",
    ),
}


class MultimodalService:
    def __init__(self, model_root: str = "models") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_root = Path(model_root)
        self._text_tokenizers: Dict[str, AutoTokenizer] = {}
        self._text_models: Dict[str, AutoModelForSequenceClassification] = {}
        self._audio_extractors: Dict[str, AutoFeatureExtractor] = {}
        self._audio_models: Dict[str, AutoModelForAudioClassification] = {}
        self._meta_models: Dict[str, Optional[object]] = {}

    def supported_languages(self) -> list[str]:
        return list(LANGUAGE_CONFIGS.keys())

    def _language_config(self, language: str) -> LanguageConfig:
        normalized = language.strip().lower()
        if normalized not in LANGUAGE_CONFIGS:
            raise ValueError(f"Unsupported language: {language}")
        return LANGUAGE_CONFIGS[normalized]

    def _get_text_stack(self, language: str):
        if language not in self._text_models:
            cfg = self._language_config(language)
            tokenizer = AutoTokenizer.from_pretrained(cfg.text_model)
            model = AutoModelForSequenceClassification.from_pretrained(cfg.text_model).to(self.device)
            model.eval()
            self._text_tokenizers[language] = tokenizer
            self._text_models[language] = model
        return self._text_tokenizers[language], self._text_models[language]

    def _get_audio_stack(self, language: str):
        if language not in self._audio_models:
            cfg = self._language_config(language)
            extractor = AutoFeatureExtractor.from_pretrained(cfg.audio_model)
            model = AutoModelForAudioClassification.from_pretrained(cfg.audio_model).to(self.device)
            model.eval()
            self._audio_extractors[language] = extractor
            self._audio_models[language] = model
        return self._audio_extractors[language], self._audio_models[language]

    def _get_meta_model(self, language: str):
        if language not in self._meta_models:
            cfg = self._language_config(language)
            model_path = self.model_root / language / cfg.meta_model_file
            self._meta_models[language] = joblib.load(model_path) if model_path.exists() else None
        return self._meta_models[language]

    def _predict_text_probability(self, language: str, text: str) -> np.ndarray:
        tokenizer, model = self._get_text_stack(language)
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        if probs.shape[0] == 1:
            p1 = float(probs[0])
            return np.array([1 - p1, p1], dtype=float)
        return probs[:2]

    def _predict_audio_probability(self, language: str, audio_bytes: bytes) -> np.ndarray:
        extractor, model = self._get_audio_stack(language)
        waveform, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        encoded = extractor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            truncation=True,
            max_length=16000 * 5,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        if probs.shape[0] == 1:
            p1 = float(probs[0])
            return np.array([1 - p1, p1], dtype=float)
        return probs[:2]

    def predict(self, language: str, text: str, audio_bytes: Optional[bytes] = None) -> dict:
        lang = language.strip().lower()
        _ = self._language_config(lang)

        text_probs = self._predict_text_probability(lang, text)
        audio_probs: Optional[np.ndarray] = None
        if audio_bytes:
            audio_probs = self._predict_audio_probability(lang, audio_bytes)

        meta_model = self._get_meta_model(lang)
        if meta_model is not None and audio_probs is not None:
            fused = np.hstack([text_probs, audio_probs]).reshape(1, -1)
            hate_prob = float(meta_model.predict_proba(fused)[0][1])
            method = "meta-model"
        elif audio_probs is not None:
            hate_prob = float((text_probs[1] + audio_probs[1]) / 2)
            method = "text+audio-average"
        else:
            hate_prob = float(text_probs[1])
            method = "text-only"

        pred = 1 if hate_prob >= 0.5 else 0
        return {
            "language": lang,
            "prediction": pred,
            "label": LABELS[pred],
            "confidence": round(max(hate_prob, 1 - hate_prob), 4),
            "hate_probability": round(hate_prob, 4),
            "fusion_method": method,
            "used_audio": audio_probs is not None,
        }
