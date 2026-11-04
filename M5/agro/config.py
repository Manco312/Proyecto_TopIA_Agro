"""CONFIG -- el único lugar donde se toca algo.

Todo lo que cambia el comportamiento del sistema en producción se lee de aquí, y casi todo
puede sobrescribirse con una variable de entorno (que es como se configura un Space de HF).
Los valores por defecto son **los mismos de M3/M4**: si nadie toca nada, el sistema desplegado
responde lo que respondía el notebook.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------------------
# Rutas. RAIZ es la carpeta M5/ (la raíz del Space).
# --------------------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
ARTEFACTOS = Path(os.environ.get("AGRO_ARTEFACTOS", RAIZ / "artefactos"))
DIR_INDICE = ARTEFACTOS / "indice"
DIR_VISION = ARTEFACTOS / "vision"
DIR_LOGS = Path(os.environ.get("AGRO_LOGS", RAIZ / "logs"))
DIR_RESULTADOS = RAIZ / "resultados"

# El repo de los módulos anteriores: de ahí salen los PDF y los JSON con los que se
# CONSTRUYE el índice. En el Space no existe -- y no hace falta, porque el índice viaja
# ya construido en artefactos/.
ORIGEN_DATOS = Path(os.environ.get("AGRO_ORIGEN", RAIZ.parent / "modulos_anteriores" / "M4"))

# El adaptador LoRA de M1. En el repo vive en los módulos anteriores; en el Space se copia
# al lado de app.py (lo hace scripts/preparar_space.py).
_LORA_LOCAL = RAIZ / "mi-modelo-lora"
MODELO_LORA = str(Path(os.environ.get(
    "AGRO_LORA", _LORA_LOCAL if _LORA_LOCAL.exists() else ORIGEN_DATOS / "mi-modelo-lora")))

# --------------------------------------------------------------------------------------
# Modelos. Los mismos IDs de M1-M4.
# --------------------------------------------------------------------------------------
MODEL_BASE_ID = "Qwen/Qwen2.5-0.5B-Instruct"       # el redactor afinado en M1
PLANIFICADOR_ID = "Qwen/Qwen2.5-1.5B-Instruct"     # el "cerebro" ReAct de M3 (§14)
EMBED_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CLIP_ID = "openai/clip-vit-base-patch32"

# --------------------------------------------------------------------------------------
# Generación. Idénticos a M2/M3/M4: greedy, 200 tokens, SEED 42.
# --------------------------------------------------------------------------------------
SEED = 42
# El tope de M2, dimensionado para un 0.5B en una laptop. Se queda como está porque es el
# que midieron todos los scorecards: cambiarlo rompería la comparación con M2, M3 y M4.
MAX_NEW_SISTEMA = 200
# El tope del redactor por API, que no tiene por qué heredar la restricción de un 0.5B.
# Con 200 las respuestas se cortaban a media frase ("...variedades susceptibles (Typica,").
MAX_NEW_GROQ = int(os.environ.get("AGRO_MAX_TOKENS", "900"))
CHUNK_SIZE, OVERLAP = 400, 60
# K_FINAL y MAX_CTX_CHARS también venían dimensionados para el 0.5B. Se dejan en el valor
# medido (cambiarlos cambia lo que entra al prompt y por tanto el scorecard), pero se
# pueden subir por entorno: con un modelo por API cabe más contexto sin penalización.
K_FINAL = int(os.environ.get("AGRO_K_FINAL", "3"))
K_CANDIDATOS, RRF_K = 10, 60
MAX_CTX_CHARS = int(os.environ.get("AGRO_MAX_CTX", "3200"))
DIAG_RANKER = "cross"

# --------------------------------------------------------------------------------------
# M5 -- lo que es nuevo de esta entrega.
# --------------------------------------------------------------------------------------
# Backend del redactor.
#   fp32peft_cpu : el adaptador LoRA SIN fusionar, tal como lo entregó M1
#   fp32_cpu     : con el adaptador fusionado (merge_and_unload)      <- PRODUCCIÓN en CPU
#   bf16_cpu     : el mismo, en bfloat16
#   int8_cpu     : cuantización dinámica int8 de torch sobre nn.Linear
#   fp16_gpu     : la línea base de M1-M4 (fp16 en CUDA)
#   int4_gpu     : NF4 con bitsandbytes (la técnica del lab S15; necesita CUDA)
#   groq         : API externa, NO es el sistema del curso (ver README)
#
# En CPU se despliega `fp32_cpu` y NO una variante comprimida: int8 e int4 incumplen el
# contrato de calidad (ver resultados/eficiencia_*.md). Lo único que se "optimiza" en
# producción es la fusión del adaptador, que es exacta.
BACKEND = os.environ.get("AGRO_BACKEND", "auto")

# Ruta del sistema sin foto:
#   "react"        : el bucle ReAct de M3 §14 con el planificador de 1.5B
#   "determinista" : diagnóstico diferencial + ficha, sin LLM planificador  <- PRODUCCIÓN CPU
#   "auto"         : react si hay CUDA, determinista si no
RUTA_TEXTO = os.environ.get("AGRO_RUTA", "auto")
MAX_PASOS_REACT = 4

# Quién elige las herramientas en el bucle ReAct: "auto" usa Groq si es el redactor
# desplegado (mismo motor, mucho mejor eligiendo) y el Qwen2.5-1.5B de M3 si no.
PLANIFICADOR_MOTOR = os.environ.get("AGRO_PLANIFICADOR", "auto")

# El filtro de salida de M5 (ver agro/sistema.py). Apagarlo sólo sirve para la ablación.
FILTRO_SALIDA = os.environ.get("AGRO_FILTRO", "1") not in ("0", "false", "False")

# Observabilidad
WANDB_PROYECTO = os.environ.get("AGRO_WANDB_PROYECTO", "agro-m5-produccion")
WANDB_ACTIVO = os.environ.get("AGRO_WANDB", "0") not in ("0", "false", "False")
LOG_JSONL = os.environ.get("AGRO_LOG_JSONL", str(DIR_LOGS / "peticiones.jsonl"))

# Groq. En M3 se usó para el TERCER JUEZ (§19); en M5 es el **redactor desplegado**: la
# generación es el cuello de botella medido del sistema desde M3, y es la única pieza que
# cambiar mueve la aguja. Si no hay clave, el sistema sigue funcionando con el modelo
# propio de M1 (ver `resolver_backend`).
try:                                   # en local: M5/.env (ignorado por git), como en M3
    from dotenv import load_dotenv
    load_dotenv(RAIZ / ".env")
except Exception:
    pass
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODELO = os.environ.get("AGRO_GROQ_MODELO", "openai/gpt-oss-120b")

# Umbrales del harness, heredados de M2/M3 sin cambios (los usa el bench).
UMBRAL_SIM, UMBRAL_JUEZ, UMBRAL_CLAVE = 0.60, 4.0, 0.40

VERSION = os.environ.get("AGRO_VERSION", "m5.0.0")


def en_zerogpu() -> bool:
    """Hugging Face pone SPACES_ZERO_GPU en los Spaces con ZeroGPU."""
    return os.environ.get("SPACES_ZERO_GPU", "").strip().lower() in ("1", "true", "yes", "on")


def resolver_backend(backend: str = None) -> str:
    """'auto' -> groq si hay clave; si no, fp16_gpu con GPU y fp32_cpu sin ella.

    Groq va primero a propósito: el redactor es el cuello de botella medido del sistema
    (M3 §21, M4 §9). Sin clave no se cae nada — se usa el modelo propio de M1, que es el
    que miden todos los scorecards.

    En ZeroGPU se fuerza `fp16_gpu` sin consultar a torch: la GPU no está asignada mientras
    arranca la app, así que preguntar `torch.cuda.is_available()` en ese momento no es una
    respuesta fiable sobre lo que habrá cuando llegue una petición.
    """
    backend = backend or BACKEND
    if backend != "auto":
        return backend
    if GROQ_API_KEY:
        return "groq"
    if en_zerogpu():
        return "fp16_gpu"
    try:
        import torch
        return "fp16_gpu" if torch.cuda.is_available() else "fp32_cpu"
    except Exception:
        return "fp32_cpu"


def resolver_motor_planificador() -> str:
    """'auto' -> groq si es el redactor desplegado, local si no."""
    motor = PLANIFICADOR_MOTOR
    if motor != "auto":
        return motor
    return "groq" if resolver_backend() == "groq" else "local"


def resolver_ruta(ruta: str = None) -> str:
    """'auto' -> react si hay API o GPU, determinista si no (el planificador de 1.5B no cabe
    en el presupuesto de latencia de 2 vCPU: ver la ablación del reporte)."""
    ruta = ruta or RUTA_TEXTO
    if ruta != "auto":
        return ruta
    if GROQ_API_KEY:          # con la API, el bucle del agente sale casi gratis
        return "react"
    if en_zerogpu():
        return "react"
    try:
        import torch
        return "react" if torch.cuda.is_available() else "determinista"
    except Exception:
        return "determinista"
