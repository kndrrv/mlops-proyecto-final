"""
retrain.py — Pipeline de reentrenamiento automático.

Flujo completo:
  1. Descarga datos frescos desde S3 (via DVC pull)
  2. Entrena el modelo
  3. Evalúa métricas
  4. Compara contra modelo en producción
  5. Promueve el nuevo modelo si mejora
  6. Sube artefactos a S3 (via DVC push)
  7. Notifica al servicio para recargar el modelo
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import requests

from train import main as run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

METRICS_PATH = Path(os.getenv("METRICS_PATH", "metrics.json"))
BASELINE_METRICS_PATH = Path(os.getenv("BASELINE_METRICS_PATH", "metrics_baseline.json"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
PROMOTE_THRESHOLD = float(os.getenv("PROMOTE_THRESHOLD", "0.01"))
API_RELOAD_URL = os.getenv("API_RELOAD_URL", "http://localhost:8000/reload-model")


def _run(cmd: list[str]) -> None:
    """Ejecuta un comando en shell; lanza excepción si falla."""
    logger.info("Ejecutando: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.returncode != 0:
        logger.error(result.stderr.strip())
        raise RuntimeError(f"Comando fallido: {' '.join(cmd)}")


def pull_data() -> None:
    """Descarga datos y modelos actuales desde S3 via DVC."""
    logger.info("Descargando datos desde S3...")
    _run(["dvc", "pull", "--force"])


def push_artifacts() -> None:
    """Sube modelo y datos actualizados a S3 via DVC."""
    logger.info("Subiendo artefactos a S3...")
    _run(["dvc", "add", str(MODEL_PATH)])
    _run(["dvc", "push"])
    _run(["git", "add", f"{MODEL_PATH}.dvc"])
    _run(["git", "commit", "-m", "chore: update model artifact [skip ci]"])
    _run(["git", "push"])


def load_metrics(path: Path) -> dict:
    """Carga métricas desde JSON; retorna vacío si no existe."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def should_promote(new: dict, baseline: dict) -> bool:
    """
    Promueve el nuevo modelo si su accuracy mejora por encima del umbral
    o si no hay baseline previo.
    """
    if not baseline:
        logger.info("No hay baseline previo — promoviendo automáticamente.")
        return True
    delta = new.get("accuracy", 0) - baseline.get("accuracy", 0)
    logger.info(
        "Accuracy nueva=%.4f | baseline=%.4f | delta=%.4f | umbral=%.4f",
        new.get("accuracy", 0),
        baseline.get("accuracy", 0),
        delta,
        PROMOTE_THRESHOLD,
    )
    return delta >= PROMOTE_THRESHOLD


def notify_api_reload() -> None:
    """Llama al endpoint de la API para recargar el modelo en memoria."""
    try:
        resp = requests.post(API_RELOAD_URL, timeout=10)
        resp.raise_for_status()
        logger.info("API notificada para recargar el modelo.")
    except Exception as exc:
        logger.warning("No se pudo notificar a la API: %s", exc)


def main() -> int:
    """Retorna 0 si se promovió el modelo, 1 si no hubo mejora."""
    logger.info("=== Reentrenamiento automático iniciado ===")

    pull_data()

    baseline = load_metrics(BASELINE_METRICS_PATH)

    new_metrics = run_training()

    if should_promote(new_metrics, baseline):
        logger.info("Nuevo modelo promovido a producción.")
        # Actualizar baseline
        import shutil
        shutil.copy(METRICS_PATH, BASELINE_METRICS_PATH)
        push_artifacts()
        notify_api_reload()
        logger.info("=== Reentrenamiento finalizado: modelo promovido ===")
        return 0
    else:
        logger.info("Nuevo modelo NO supera el baseline — descartado.")
        # Restaurar el modelo anterior
        _run(["dvc", "checkout", str(MODEL_PATH)])
        logger.info("=== Reentrenamiento finalizado: sin cambios ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
