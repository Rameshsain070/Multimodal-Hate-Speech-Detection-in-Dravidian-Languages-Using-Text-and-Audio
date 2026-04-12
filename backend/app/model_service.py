from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
    text_models: Dict[str, str]
    audio_models: Dict[str, str]
    default_text_models: List[str]
    default_audio_models: List[str]
    meta_model_file: str


LANGUAGE_CONFIGS: Dict[str, LanguageConfig] = {
    "tamil": LanguageConfig(
        text_models={
            "indicbert": "ramesh070/indicbert-hatespeechdetection-tamil",
            "xlm-r": "ramesh070/xlm-r-hatespeechdetection-tamil",
            "mbert": "ramesh070/mbert-hatespeechdetection-tamil",
        },
        audio_models={
            "wav2vec2": "ramesh070/wav2vec2-tamil-binary",
            "wavlm": "ramesh070/wavlm-tamil-binary",
            "mms": "ramesh070/mms-tamil-binary",
        },
        default_text_models=["xlm-r"],
        default_audio_models=["wav2vec2"],
        meta_model_file="meta_lightgbm_tamil_from_eval_safe.joblib",
    ),
    "telugu": LanguageConfig(
        text_models={
            "indicbert": "ramesh070/indicbert-hatespeechdetection-telugu",
            "xlm-r": "ramesh070/xlm-r-hatespeechdetection-telugu",
            "mbert": "ramesh070/mbert-hatespeechdetection-telugu",
        },
        audio_models={
            "wav2vec2": "ramesh070/wav2vec2-telugu-binary",
            "wavlm": "ramesh070/wavlm-telugu-binary",
            "mms": "ramesh070/mms-telugu-binary",
        },
        default_text_models=["xlm-r"],
        default_audio_models=["wav2vec2"],
        meta_model_file="meta_lightgbm_telugu_from_eval_safe.joblib",
    ),
    "malayalam": LanguageConfig(
        text_models={
            "indicbert": "ramesh070/indicbert-hatespeechdetection-malayalam",
            "xlm-r": "ramesh070/xlm-r-hatespeechdetection-malayalam",
            "mbert": "ramesh070/mbert-hatespeechdetection-malayalam",
        },
        audio_models={
            "wav2vec2": "ramesh070/wav2vec2-malayalam-binary",
            "wavlm": "ramesh070/wavlm-malayalam-binary",
            "mms": "ramesh070/mms-malayalam-binary",
        },
        default_text_models=["xlm-r"],
        default_audio_models=["wav2vec2"],
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

    def supported_languages(self) -> List[str]:
        return list(LANGUAGE_CONFIGS.keys())

    def _language_config(self, language: str) -> LanguageConfig:
        normalized = language.strip().lower()
        if normalized not in LANGUAGE_CONFIGS:
            raise ValueError(f"Unsupported language: {language}")
        return LANGUAGE_CONFIGS[normalized]

    def _text_cache_key(self, language: str, model_key: str) -> str:
        return f"{language}:{model_key}"

    def _audio_cache_key(self, language: str, model_key: str) -> str:
        return f"{language}:{model_key}"

    def model_catalog(self) -> Dict[str, dict]:
        catalog: Dict[str, dict] = {}
        for language, cfg in LANGUAGE_CONFIGS.items():
            catalog[language] = {
                "text_models": cfg.text_models,
                "audio_models": cfg.audio_models,
                "default_text_models": cfg.default_text_models,
                "default_audio_models": cfg.default_audio_models,
            }
        return catalog

    def resolve_model_selection(
        self,
        language: str,
        text_model_keys: Optional[List[str]] = None,
        audio_model_keys: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        cfg = self._language_config(language)

        def _normalize_model_selection(
            selected: Optional[List[str]],
            available: Dict[str, str],
            defaults: List[str],
        ) -> List[str]:
            if selected is None:
                return list(defaults)
            normalized = [item.strip().lower() for item in selected if item and item.strip()]
            if not normalized:
                return []
            if "default" in normalized:
                return list(defaults)
            if "all" in normalized:
                return list(available.keys())
            unknown = [item for item in normalized if item not in available]
            if unknown:
                raise ValueError(f"Unsupported model key(s): {', '.join(sorted(set(unknown)))}")
            deduped = []
            seen = set()
            for item in normalized:
                if item not in seen:
                    seen.add(item)
                    deduped.append(item)
            return deduped

        return {
            "text": _normalize_model_selection(text_model_keys, cfg.text_models, cfg.default_text_models),
            "audio": _normalize_model_selection(audio_model_keys, cfg.audio_models, cfg.default_audio_models),
        }

    def _get_text_stack(self, language: str, model_key: str):
        cfg = self._language_config(language)
        if model_key not in cfg.text_models:
            raise ValueError(f"Unsupported text model '{model_key}' for language '{language}'.")
        cache_key = self._text_cache_key(language, model_key)
        if cache_key not in self._text_models:
            model_id = cfg.text_models[model_key]
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)
            model.eval()
            self._text_tokenizers[cache_key] = tokenizer
            self._text_models[cache_key] = model
        return self._text_tokenizers[cache_key], self._text_models[cache_key]

    def _get_audio_stack(self, language: str, model_key: str):
        cfg = self._language_config(language)
        if model_key not in cfg.audio_models:
            raise ValueError(f"Unsupported audio model '{model_key}' for language '{language}'.")
        cache_key = self._audio_cache_key(language, model_key)
        if cache_key not in self._audio_models:
            model_id = cfg.audio_models[model_key]
            extractor = AutoFeatureExtractor.from_pretrained(model_id)
            model = AutoModelForAudioClassification.from_pretrained(model_id).to(self.device)
            model.eval()
            self._audio_extractors[cache_key] = extractor
            self._audio_models[cache_key] = model
        return self._audio_extractors[cache_key], self._audio_models[cache_key]

    def _get_meta_model(self, language: str):
        if language not in self._meta_models:
            cfg = self._language_config(language)
            model_path = self.model_root / language / cfg.meta_model_file
            self._meta_models[language] = joblib.load(model_path) if model_path.exists() else None
        return self._meta_models[language]

    def _predict_text_probability(self, language: str, model_key: str, text: str) -> np.ndarray:
        tokenizer, model = self._get_text_stack(language, model_key)
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        if probs.shape[0] == 1:
            p1 = float(probs[0])
            return np.array([1 - p1, p1], dtype=float)
        return probs[:2]

    def _predict_audio_probability(self, language: str, model_key: str, audio_bytes: bytes) -> np.ndarray:
        extractor, model = self._get_audio_stack(language, model_key)
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

    def warmup(self, preload_all: bool = False) -> dict:
        warmed = {"text": 0, "audio": 0, "meta": 0}
        for language, cfg in LANGUAGE_CONFIGS.items():
            text_keys = cfg.text_models.keys() if preload_all else cfg.default_text_models
            audio_keys = cfg.audio_models.keys() if preload_all else cfg.default_audio_models
            for model_key in text_keys:
                self._get_text_stack(language, model_key)
                warmed["text"] += 1
            for model_key in audio_keys:
                self._get_audio_stack(language, model_key)
                warmed["audio"] += 1
            _ = self._get_meta_model(language)
            warmed["meta"] += 1
        return warmed

    def predict(
        self,
        language: str,
        text: Optional[str] = None,
        audio_bytes: Optional[bytes] = None,
        text_model_keys: Optional[List[str]] = None,
        audio_model_keys: Optional[List[str]] = None,
    ) -> dict:
        lang = language.strip().lower()
        _ = self._language_config(lang)
        has_text = bool(text and text.strip())
        has_audio = bool(audio_bytes)
        if not has_text and not has_audio:
            raise ValueError("At least one input is required: text or audio.")

        selected = self.resolve_model_selection(
            language=lang,
            text_model_keys=text_model_keys,
            audio_model_keys=audio_model_keys,
        )
        if has_text and not selected["text"]:
            raise ValueError("No text models selected for text input.")
        if has_audio and not selected["audio"]:
            raise ValueError("No audio models selected for audio input.")

        text_outputs: Dict[str, np.ndarray] = {}
        if has_text:
            cleaned_text = text.strip()
            for model_key in selected["text"]:
                text_outputs[model_key] = self._predict_text_probability(lang, model_key, cleaned_text)

        audio_outputs: Dict[str, np.ndarray] = {}
        if has_audio and audio_bytes is not None:
            for model_key in selected["audio"]:
                audio_outputs[model_key] = self._predict_audio_probability(lang, model_key, audio_bytes)

        text_probs: Optional[np.ndarray] = None
        if text_outputs:
            text_probs = np.mean(np.array(list(text_outputs.values())), axis=0)
        audio_probs: Optional[np.ndarray] = None
        if audio_outputs:
            audio_probs = np.mean(np.array(list(audio_outputs.values())), axis=0)

        fusion_method = "unknown"
        fusion_warning: Optional[str] = None
        if text_probs is not None and audio_probs is not None:
            meta_model = self._get_meta_model(lang)
            if meta_model is not None:
                try:
                    fused = np.hstack([text_probs, audio_probs]).reshape(1, -1)
                    hate_prob = float(meta_model.predict_proba(fused)[0][1])
                    fusion_method = "meta-model"
                except (AttributeError, IndexError, TypeError, ValueError) as exc:
                    hate_prob = float((text_probs[1] + audio_probs[1]) / 2)
                    fusion_method = "text-audio-mean-fallback"
                    fusion_warning = f"Meta-model fallback used: {exc}"
            else:
                hate_prob = float((text_probs[1] + audio_probs[1]) / 2)
                fusion_method = "text-audio-mean"
        elif text_probs is not None:
            hate_prob = float(text_probs[1])
            fusion_method = "text-only"
        elif audio_probs is not None:
            hate_prob = float(audio_probs[1])
            fusion_method = "audio-only"
        else:
            raise ValueError("No model outputs were generated.")

        pred = 1 if hate_prob >= 0.5 else 0
        model_outputs = {
            "text": {
                model_key: {
                    "hate_probability": round(float(prob[1]), 4),
                    "confidence": round(float(max(prob[0], prob[1])), 4),
                    "label": LABELS[1 if float(prob[1]) >= 0.5 else 0],
                }
                for model_key, prob in text_outputs.items()
            },
            "audio": {
                model_key: {
                    "hate_probability": round(float(prob[1]), 4),
                    "confidence": round(float(max(prob[0], prob[1])), 4),
                    "label": LABELS[1 if float(prob[1]) >= 0.5 else 0],
                }
                for model_key, prob in audio_outputs.items()
            },
        }

        response = {
            "language": lang,
            "prediction": pred,
            "label": LABELS[pred],
            "confidence": round(max(hate_prob, 1 - hate_prob), 4),
            "hate_probability": round(hate_prob, 4),
            "fusion_method": fusion_method,
            "used_text": text_probs is not None,
            "used_audio": audio_probs is not None,
            "selected_models": selected,
            "model_outputs": model_outputs,
        }
        if fusion_warning is not None:
            response["fusion_warning"] = fusion_warning
        return response
