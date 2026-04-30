"""
predict.py — Lógica de inferencia para el clasificador de intenciones.

Recibe texto libre y devuelve la intención predicha con su probabilidad.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODEL_PATH   = Path(os.getenv("MODEL_PATH",   "models/model.pkl"))
ENCODER_PATH = Path(os.getenv("ENCODER_PATH", "models/label_encoder.pkl"))

_pipeline_cache = None
_encoder_cache  = None


def load_artifacts(
    model_path: Path = MODEL_PATH,
    encoder_path: Path = ENCODER_PATH,
):
    """Carga pipeline y encoder desde disco. Usa caché en memoria."""
    global _pipeline_cache, _encoder_cache
    if _pipeline_cache is None:
        logger.info("Cargando pipeline desde %s", model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
        with open(model_path, "rb") as f:
            _pipeline_cache = pickle.load(f)

        logger.info("Cargando encoder desde %s", encoder_path)
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder no encontrado en {encoder_path}")
        with open(encoder_path, "rb") as f:
            _encoder_cache = pickle.load(f)

        logger.info("Artefactos cargados. Clases: %s", list(_encoder_cache.classes_))
    return _pipeline_cache, _encoder_cache


def predict(text: str) -> dict[str, Any]:
    """
    Clasifica la intención de un texto.

    Args:
        text: Frase del usuario en texto libre.

    Returns:
        {
          "intent":        nombre de la intención predicha,
          "confidence":    probabilidad de la clase ganadora (0-1),
          "all_intents":   lista de {intent, confidence} ordenada de mayor a menor,
        }
    """
    pipeline, encoder = load_artifacts()

    proba = pipeline.predict_proba([text])[0]
    pred_idx = proba.argmax()

    all_intents = [
        {"intent": encoder.classes_[i], "confidence": round(float(p), 4)}
        for i, p in enumerate(proba)
    ]
    all_intents.sort(key=lambda x: x["confidence"], reverse=True)

    result = {
        "intent":      encoder.classes_[pred_idx],
        "confidence":  round(float(proba[pred_idx]), 4),
        "all_intents": all_intents,
    }
    logger.info("Texto: '%s' → intención: '%s' (%.2f%%)",
                text, result["intent"], result["confidence"] * 100)
    return result


def reload_model() -> None:
    """Fuerza la recarga de artefactos (útil tras reentrenamiento)."""
    global _pipeline_cache, _encoder_cache
    _pipeline_cache = None
    _encoder_cache  = None
    load_artifacts()
    logger.info("Artefactos recargados en memoria.")
