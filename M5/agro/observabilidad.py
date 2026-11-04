"""Producción verificable: health check, una línea JSON por petición, y W&B.

Las tres cosas que la rúbrica de M5 llama "producción verificable", y que son las tres que
permiten responder "¿está vivo?", "¿qué hizo con mi consulta?" y "¿cómo se está portando?"
sin abrir el notebook.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import threading
import time
from collections import deque
from datetime import datetime, timezone

from . import config

_T0 = time.time()
_lock = threading.Lock()
_estado = {
    "peticiones": 0, "errores": 0, "con_foto": 0, "valvula": 0, "filtro": 0,
    "latencias": deque(maxlen=500), "ultimo_error": None, "arranque": None,
}
_wandb = {"run": None, "activo": False}


def _pct(valores, q):
    if not valores:
        return None
    v = sorted(valores)
    i = min(len(v) - 1, max(0, int(round((q / 100) * (len(v) - 1)))))
    return round(v[i], 3)


def _sha256(path, tope=8):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for bloque in iter(lambda: f.read(1 << 20), b""):
                h.update(bloque)
        return h.hexdigest()[:tope]
    except OSError:
        return None


# --------------------------------------------------------------------------------------
# Logging estructurado: una línea JSON por petición
# --------------------------------------------------------------------------------------
def registrar(pregunta, detalle, error=None):
    """Escribe la línea de log y actualiza los contadores. No lanza nunca: si el logging
    falla, la respuesta al agricultor ya salió y no se va a perder por eso."""
    linea = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": config.VERSION,
        "evento": "error" if error else "peticion",
        "entrada": {"caracteres": len(pregunta or ""), "con_foto": bool(detalle.get("con_foto")),
                    "cultivos_detectados": detalle.get("cultivos_detectados")},
        "ruta": detalle.get("ruta"),
        "backend": detalle.get("backend"),
        "planificador": detalle.get("planificador"),
        "confianza": {"p_top1": detalle.get("p_top1"), "cos_prototipo": detalle.get("cos_prototipo"),
                      "p_hoja": detalle.get("p_hoja"), "etiqueta": detalle.get("etiqueta_pred")},
        "valvula": {"disparo": detalle.get("valvula"), "motivos": detalle.get("motivos")},
        "filtro_salida": {k: v for k, v in (detalle.get("filtro") or {}).items()
                          if k != "frases_retiradas"},
        "latencia_s": {"total": detalle.get("latencia_total_s"),
                       "vision": detalle.get("latencia_vision_s")},
        "tokens_generados": detalle.get("tokens_generados"),
    }
    if error:
        linea["error"] = f"{type(error).__name__}: {error}"[:300]

    with _lock:
        _estado["peticiones"] += 1
        if error:
            _estado["errores"] += 1
            _estado["ultimo_error"] = linea["error"]
        if detalle.get("con_foto"):
            _estado["con_foto"] += 1
        if detalle.get("valvula"):
            _estado["valvula"] += 1
        if (detalle.get("filtro") or {}).get("n_frases_retiradas"):
            _estado["filtro"] += 1
        # La latencia de una petición que falló no es una latencia de servicio: meterla en
        # la serie hunde el p50 y el p95 y hace parecer rápido a un servicio roto.
        if not error and detalle.get("latencia_total_s"):
            _estado["latencias"].append(float(detalle["latencia_total_s"]))

    try:
        os.makedirs(os.path.dirname(config.LOG_JSONL) or ".", exist_ok=True)
        with open(config.LOG_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(linea, ensure_ascii=False) + "\n")
    except OSError:
        pass

    _a_wandb(linea)
    return linea


# --------------------------------------------------------------------------------------
# Weights & Biases: las cuatro gráficas de la S15 (latencia, ruta, confianza, válvula/filtro)
# --------------------------------------------------------------------------------------
def iniciar_wandb(forzar=False):
    if _wandb["activo"]:
        return True
    if not (config.WANDB_ACTIVO or forzar) or not os.environ.get("WANDB_API_KEY"):
        return False
    try:
        import wandb
        _wandb["run"] = wandb.init(
            project=config.WANDB_PROYECTO, job_type="produccion",
            name=f"space-{config.VERSION}-{datetime.now():%Y%m%d-%H%M}",
            config={"version": config.VERSION, "backend": config.resolver_backend(),
                    "ruta": config.resolver_ruta(), "filtro_salida": config.FILTRO_SALIDA},
            reinit=True)
        _wandb["activo"] = True
    except Exception:
        _wandb["activo"] = False
    return _wandb["activo"]


def _a_wandb(linea):
    if not _wandb["activo"]:
        return
    try:
        import wandb
        lat = linea["latencia_s"]
        wandb.log({
            "latencia/total_s": lat["total"], "latencia/vision_s": lat["vision"],
            "latencia/p50_s": _pct(list(_estado["latencias"]), 50),
            "latencia/p95_s": _pct(list(_estado["latencias"]), 95),
            "ruta/" + str(linea["ruta"]): 1,
            "confianza/p_top1": linea["confianza"]["p_top1"],
            "confianza/cos_prototipo": linea["confianza"]["cos_prototipo"],
            "guardas/valvula": int(bool(linea["valvula"]["disparo"])),
            "guardas/filtro_salida": int(bool(linea["filtro_salida"].get("n_frases_retiradas"))),
            "tokens_generados": linea["tokens_generados"],
            "errores_acumulados": _estado["errores"],
        })
    except Exception:
        pass


def cerrar_wandb():
    if _wandb["activo"]:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
        _wandb["activo"] = False


# --------------------------------------------------------------------------------------
# Health check
# --------------------------------------------------------------------------------------
def marcar_arranque(segundos, piezas):
    _estado["arranque"] = {"segundos": round(segundos, 2), "piezas": piezas}


def _info_redactor():
    """Qué está redactando de verdad, y si la API está respondiendo."""
    try:
        from .modelos import _cache
        for clave, r in _cache.items():
            if clave.startswith("redactor:"):
                return r.info()
    except Exception:
        pass
    return {"backend": config.resolver_backend(), "cargado": False}


def _umbrales_vision():
    """El punto de operación de la válvula, y si se movió respecto del calibrado en M4."""
    try:
        from . import vision
        e = vision.cargar()
        return {"umbrales_operativos": e["umbrales_operativos"],
                "umbrales_calibrados_m4": e["umbrales_calibrados"],
                "umbrales_movidos": e["umbrales_movidos"] or None}
    except Exception:
        return {}


def salud(detallado=True):
    """Lo que debe poder leerse sin entrar al código: versión, qué está cargado, cuánto
    tarda y cuántas veces se han disparado las guardas."""
    from . import indice
    lat = list(_estado["latencias"])
    try:
        n_chunks = len(indice.chunks())
    except Exception:
        n_chunks = None

    estado = "ok"
    if _estado["arranque"] is None:
        estado = "arrancando"
    elif _estado["errores"] and _estado["errores"] / max(1, _estado["peticiones"]) > 0.2:
        estado = "degradado"

    salida = {
        "estado": estado,
        "version": config.VERSION,
        "uptime_s": round(time.time() - _T0, 1),
        "arranque": _estado["arranque"],
        "configuracion": {
            "backend": config.resolver_backend(),
            "planificador": config.resolver_ruta(),
            "filtro_salida": config.FILTRO_SALIDA,
            "max_new_tokens": config.MAX_NEW_SISTEMA,
            "k_final": config.K_FINAL, "k_candidatos": config.K_CANDIDATOS,
        },
        "indice": {"chunks": n_chunks,
                   "sha256_embeddings": _sha256(config.DIR_INDICE / "embeddings.npy"),
                   "sha256_catalogo": _sha256(config.DIR_INDICE / "catalogo.json")},
        "vision": {"sha256_clasificador": _sha256(config.DIR_VISION / "clasificador_m4.joblib"),
                   "sha256_prototipos": _sha256(config.DIR_VISION / "prototipos.npy"),
                   **_umbrales_vision()},
        "trafico": {"peticiones": _estado["peticiones"], "errores": _estado["errores"],
                    "con_foto": _estado["con_foto"], "valvula_disparada": _estado["valvula"],
                    "filtro_salida_activado": _estado["filtro"]},
        "latencia_s": {"n": len(lat), "p50": _pct(lat, 50), "p95": _pct(lat, 95),
                       "max": round(max(lat), 3) if lat else None},
        "redactor": _info_redactor(),
        "wandb": {"activo": _wandb["activo"], "proyecto": config.WANDB_PROYECTO},
        "ultimo_error": _estado["ultimo_error"],
    }
    if detallado:
        salida["entorno"] = {
            "python": platform.python_version(), "sistema": platform.system(),
            "cpus": os.cpu_count(),
        }
        try:
            import torch
            salida["entorno"]["torch"] = torch.__version__
            salida["entorno"]["cuda"] = torch.cuda.is_available()
        except Exception:
            pass
    return salida
