"""
train.py — Entrenamiento del clasificador de intenciones (chatbot).

Pipeline: TfidfVectorizer → LogisticRegression
Guarda:
  - models/model.pkl         (pipeline sklearn completo)
  - models/label_encoder.pkl (mapeo índice <-> nombre de intención)
  - metrics.json             (métricas por clase + global)
"""

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DATA_PATH    = Path(os.getenv("DATA_PATH",    "data/dataset.csv"))
MODEL_PATH   = Path(os.getenv("MODEL_PATH",   "models/model.pkl"))
ENCODER_PATH = Path(os.getenv("ENCODER_PATH", "models/label_encoder.pkl"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "metrics.json"))
TEXT_COLUMN   = os.getenv("TEXT_COLUMN",   "text")
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "intent")
TEST_SIZE     = float(os.getenv("TEST_SIZE",  "0.2"))
RANDOM_STATE  = int(os.getenv("RANDOM_STATE", "42"))
MAX_ITER      = int(os.getenv("MAX_ITER",     "1000"))


def load_data(path: Path) -> pd.DataFrame:
    """Carga el dataset desde disco."""
    logger.info("Cargando datos desde %s", path)
    df = pd.read_csv(path)
    logger.info("Dataset: %d ejemplos, %d intenciones", len(df), df[TARGET_COLUMN].nunique())
    return df


def preprocess(df: pd.DataFrame):
    """Codifica labels y hace split train/test estratificado."""
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET_COLUMN])
    X = df[TEXT_COLUMN].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Clases: %s", list(encoder.classes_))
    logger.info("Split: %d train | %d test", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test, encoder


def build_pipeline() -> Pipeline:
    """Construye el pipeline TF-IDF + Logistic Regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{1,}",
        )),
        ("clf", LogisticRegression(
            max_iter=MAX_ITER,
            C=float(os.getenv("C", "5.0")),
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )),
    ])


def train_model(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    """Entrena el pipeline completo."""
    logger.info("Entrenando...")
    pipeline.fit(X_train, y_train)
    logger.info("Entrenamiento completo.")
    return pipeline


def evaluate(pipeline: Pipeline, X_test, y_test, encoder: LabelEncoder) -> dict:
    """Evalúa el modelo y retorna métricas globales + por clase."""
    preds = pipeline.predict(X_test)
    class_names = list(encoder.classes_)
    report = classification_report(y_test, preds, target_names=class_names, output_dict=True)
    metrics = {
        "accuracy":  round(accuracy_score(y_test, preds), 4),
        "f1":        round(f1_score(y_test, preds, average="weighted"), 4),
        "precision": round(precision_score(y_test, preds, average="weighted"), 4),
        "recall":    round(recall_score(y_test, preds, average="weighted"), 4),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"], 4),
                "f1":        round(report[cls]["f1-score"], 4),
                "support":   int(report[cls]["support"]),
            }
            for cls in class_names
        },
        "trained_at":  datetime.utcnow().isoformat(),
        "num_classes": len(class_names),
        "classes":     class_names,
    }
    logger.info("Accuracy: %.4f | F1: %.4f", metrics["accuracy"], metrics["f1"])
    return metrics


def save_artifact(obj, path: Path) -> None:
    """Serializa un objeto Python a disco con pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Artefacto guardado en %s", path)


def save_metrics(metrics: dict, path: Path) -> None:
    """Persiste las métricas en JSON (compatible con DVC)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Métricas guardadas en %s", path)


def main() -> dict:
    logger.info("=== Pipeline de entrenamiento (Intent Classification) iniciado ===")
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, encoder = preprocess(df)
    pipeline = build_pipeline()
    pipeline = train_model(pipeline, X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test, encoder)
    save_artifact(pipeline, MODEL_PATH)
    save_artifact(encoder, ENCODER_PATH)
    save_metrics(metrics, METRICS_PATH)
    logger.info("=== Finalizado — Accuracy: %.4f ===", metrics["accuracy"])
    return metrics


if __name__ == "__main__":
    main()
