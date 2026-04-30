"""
main.py — API híbrida: TF-IDF clasifica intención + Llama3 (Groq) genera respuesta.

Flujo:
  1. /chat recibe el texto del usuario
  2. El modelo local (TF-IDF) clasifica la intención
  3. Se construye un prompt con la intención + texto original
  4. Llama3 via Groq genera una respuesta natural (gratis)
  5. Se retorna respuesta + metadatos MLOps

Endpoints:
  GET  /health        — Liveness check
  POST /chat          — Chatbot híbrido (modelo local + Groq)
  POST /predict       — Clasificación raw (para el pipeline MLOps)
  POST /reload-model  — Recarga artefactos tras reentrenamiento
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import predict, reload_model, load_artifacts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Cliente Groq (compatible con la API de OpenAI) ────────────────────────────
# Groq es gratuito y usa el mismo cliente de openai apuntando a otra base_url
groq_client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)
LLM_MODEL        = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_MAX_TOKENS   = int(os.getenv("LLM_MAX_TOKENS", "300"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))

# ── System prompt base del asistente ─────────────────────────────────────────
SYSTEM_PROMPT = """Eres un asistente virtual amable y profesional de servicio al cliente.
Tu objetivo es ayudar al usuario de forma clara, concisa y empática.

Recibirás:
- La intención detectada automáticamente del mensaje del usuario
- El mensaje original del usuario

Reglas:
- Responde SIEMPRE en español
- Sé directo y útil, máximo 3 oraciones
- Si la intención es 'desconocida', pide amablemente que reformule
- No menciones que usas inteligencia artificial ni que detectaste una intención
- Adapta el tono al contexto: formal para soporte, cálido para saludos
"""

# ── Contexto por intención (guía al LLM sin limitarlo) ───────────────────────
INTENT_CONTEXT: dict[str, str] = {
    "consultar_precio":     "El usuario pregunta sobre precios o planes. Menciona que hay planes desde $9.99/mes y ofrece más detalles.",
    "cancelar_suscripcion": "El usuario quiere cancelar. Explica el proceso (Configuración → Cuenta → Cancelar) y pregunta si puede ayudar a retenerlo.",
    "soporte_tecnico":      "El usuario tiene un problema técnico. Muestra empatía, pide más detalles del error y ofrece soluciones comunes.",
    "actualizar_datos":     "El usuario quiere cambiar datos de su perfil. Dirige a Configuración → Perfil y pregunta qué necesita cambiar.",
    "informacion_general":  "El usuario quiere información general del servicio. Invítalo a preguntar lo que necesite.",
    "hacer_reserva":        "El usuario quiere hacer una reserva. Pide fecha, hora y número de personas.",
    "rastrear_pedido":      "El usuario quiere saber dónde está su pedido. Pide el número de orden para rastrearlo.",
    "saludo":               "El usuario saluda. Responde cordialmente y pregunta en qué puedes ayudarle.",
    "despedida":            "El usuario se despide. Despídete cordialmente y deséale un buen día.",
    "desconocida":          "No se pudo determinar la intención. Pide amablemente que reformule su consulta.",
}


def build_user_prompt(text: str, intent: str, confidence: float) -> str:
    """Construye el prompt para el LLM con intención + contexto + mensaje original."""
    context = INTENT_CONTEXT.get(intent, INTENT_CONTEXT["desconocida"])
    return f"""Intención detectada: {intent} (confianza: {confidence:.0%})
Contexto de esta intención: {context}

Mensaje original del usuario: "{text}"

Genera una respuesta apropiada."""


def call_llm(user_prompt: str) -> str:
    """Llama a Llama3 via Groq y retorna el texto de respuesta."""
    logger.info("Llamando a %s via Groq...", LLM_MODEL)
    completion = groq_client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        temperature=0.7,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )
    response_text = completion.choices[0].message.content.strip()
    logger.info("Groq respondió: %s", response_text[:80])
    return response_text


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Cargando artefactos al iniciar...")
    try:
        load_artifacts()
    except FileNotFoundError as e:
        logger.warning("Artefactos no encontrados al arrancar: %s", e)
    yield
    logger.info("Apagando API.")


app = FastAPI(
    title="Chatbot Híbrido — TF-IDF + Llama3 (Groq)",
    description="Clasificación de intenciones con modelo local + generación de respuestas con Groq (gratis).",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500,
                      example="No puedo iniciar sesión en mi cuenta")


class IntentResult(BaseModel):
    intent: str
    confidence: float


class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float
    all_intents: list[IntentResult]
    model_used: str


class PredictRequest(BaseModel):
    features: dict = Field(..., example={"text": "quiero cancelar mi cuenta"})


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_model: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Infra"])
def health():
    """Liveness probe — retorna 200 si el servicio está activo."""
    return {"status": "ok", "version": app.version, "llm_model": LLM_MODEL}


@app.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
def chat(body: ChatRequest):
    """
    Chatbot híbrido.
    1. Modelo local (TF-IDF) detecta la intención.
    2. Llama3 via Groq genera una respuesta natural usando esa intención como contexto.
    """
    try:
        result     = predict(body.text)
        intent     = result["intent"]
        confidence = result["confidence"]

        if confidence < CONFIDENCE_THRESHOLD:
            intent = "desconocida"

        user_prompt   = build_user_prompt(body.text, intent, confidence)
        response_text = call_llm(user_prompt)

        return {
            "response":    response_text,
            "intent":      intent,
            "confidence":  confidence,
            "all_intents": result["all_intents"],
            "model_used":  LLM_MODEL,
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Error en /chat: %s", e)
        raise HTTPException(status_code=500, detail="Error interno al procesar la consulta.")


@app.post("/predict", tags=["MLOps"])
def predict_endpoint(body: PredictRequest):
    """
    Clasificación raw — compatible con el pipeline de reentrenamiento.
    No llama al LLM, solo retorna la intención del modelo local.
    """
    text = body.features.get("text", "")
    if not text:
        raise HTTPException(status_code=422, detail="El campo 'text' es requerido en features.")
    try:
        return predict(text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Error en /predict: %s", e)
        raise HTTPException(status_code=500, detail="Error interno.")


@app.post("/reload-model", tags=["Infra"])
def reload_model_endpoint():
    """Recarga los artefactos desde disco tras reentrenamiento."""
    try:
        reload_model()
        return {"message": "Artefactos recargados exitosamente."}
    except Exception as e:
        logger.error("Error al recargar: %s", e)
        raise HTTPException(status_code=500, detail=str(e))