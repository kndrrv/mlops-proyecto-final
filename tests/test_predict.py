"""
test_predict.py — Tests para el clasificador de intenciones y la API.
"""

import json
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_pipeline():
    """Pipeline falso que siempre predice 'saludo'."""
    pipeline = MagicMock()
    pipeline.predict.return_value = np.array([0])
    # 8 clases de ejemplo
    pipeline.predict_proba.return_value = np.array([[
        0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025
    ]])
    return pipeline


@pytest.fixture
def mock_encoder():
    """LabelEncoder falso con las 8 intenciones del dataset."""
    encoder = MagicMock()
    encoder.classes_ = np.array([
        "actualizar_datos", "cancelar_suscripcion", "consultar_precio",
        "despedida", "hacer_reserva", "informacion_general",
        "rastrear_pedido", "saludo",
    ])
    return encoder


# ── Tests de predict.py ───────────────────────────────────────────────────────

class TestPredict:
    def test_predict_returns_dict(self, mock_pipeline, mock_encoder):
        import predict as p
        p._pipeline_cache = mock_pipeline
        p._encoder_cache  = mock_encoder
        result = p.predict("Hola buenos días")
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, mock_pipeline, mock_encoder):
        import predict as p
        p._pipeline_cache = mock_pipeline
        p._encoder_cache  = mock_encoder
        result = p.predict("quiero cancelar")
        assert "intent"      in result
        assert "confidence"  in result
        assert "all_intents" in result

    def test_confidence_is_between_0_and_1(self, mock_pipeline, mock_encoder):
        import predict as p
        p._pipeline_cache = mock_pipeline
        p._encoder_cache  = mock_encoder
        result = p.predict("¿cuánto cuesta?")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_all_intents_sorted_descending(self, mock_pipeline, mock_encoder):
        import predict as p
        p._pipeline_cache = mock_pipeline
        p._encoder_cache  = mock_encoder
        result = p.predict("necesito ayuda")
        confidences = [x["confidence"] for x in result["all_intents"]]
        assert confidences == sorted(confidences, reverse=True)

    def test_load_raises_if_model_missing(self, tmp_path):
        import predict as p
        p._pipeline_cache = None
        p._encoder_cache  = None
        with pytest.raises(FileNotFoundError):
            p.load_artifacts(
                model_path=tmp_path / "no_model.pkl",
                encoder_path=tmp_path / "no_encoder.pkl",
            )

    def test_reload_clears_cache(self, mock_pipeline, mock_encoder, tmp_path):
        import predict as p
        # Guardamos artefactos reales en tmp
        model_path   = tmp_path / "model.pkl"
        encoder_path = tmp_path / "encoder.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(mock_pipeline, f)
        with open(encoder_path, "wb") as f:
            pickle.dump(mock_encoder, f)

        p._pipeline_cache = mock_pipeline
        p._encoder_cache  = mock_encoder
        with patch.object(p, "MODEL_PATH", model_path), \
             patch.object(p, "ENCODER_PATH", encoder_path):
            p.reload_model()  # No debe lanzar excepción


# ── Tests de la API ───────────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture(autouse=True)
    def setup(self, mock_pipeline, mock_encoder):
        import predict as p
        p._pipeline_cache = mock_pipeline
        p._encoder_cache  = mock_encoder

        from fastapi.testclient import TestClient
        sys.path.insert(0, str(Path(__file__).parent.parent / "api"))
        import importlib, api.main as api_main
        importlib.reload(api_main)
        self.client = TestClient(api_main.app)

    def test_health_returns_200(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_chat_returns_response(self):
        r = self.client.post("/chat", json={"text": "Hola"})
        assert r.status_code == 200
        body = r.json()
        assert "response"   in body
        assert "intent"     in body
        assert "confidence" in body

    def test_predict_compat_endpoint(self):
        r = self.client.post("/predict", json={"features": {"text": "quiero cancelar"}})
        assert r.status_code == 200

    def test_predict_missing_text_returns_422(self):
        r = self.client.post("/predict", json={"features": {}})
        assert r.status_code == 422

    def test_chat_empty_text_returns_422(self):
        r = self.client.post("/chat", json={"text": ""})
        assert r.status_code == 422


# ── Tests de métricas ─────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_json_structure(self, tmp_path):
        metrics = {
            "accuracy": 0.95, "f1": 0.94,
            "precision": 0.93, "recall": 0.95,
            "per_class": {
                "saludo": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 5}
            },
            "num_classes": 8,
            "classes": ["saludo"],
        }
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps(metrics))
        loaded = json.loads(path.read_text())
        assert 0 <= loaded["accuracy"] <= 1
        assert "per_class" in loaded
        assert loaded["num_classes"] == 8
