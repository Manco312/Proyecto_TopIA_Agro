"""`ver()` -- el componente visual de M4, con los umbrales calibrados en su corrida.

CLIP ViT-B/32 congelado + cabeza lineal de 38 clases (scikit-learn) + la válvula de cuatro
guardas. Los artefactos (`clasificador_m4.joblib`, `prototipos.npy`, `vision_config.json`)
son **los mismos bytes** que entregó M4: aquí no se reentrena ni se recalibra nada.

Nota de despliegue: `vision_config.json` fija `scikit-learn` a la versión con la que se
guardó el `.joblib`; `requirements.txt` la repite. Con otra versión el clasificador puede
no cargar o cambiar de resultado sin avisar.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .config import DIR_VISION
from .catalogo import etiquetas_pv

CULTIVO_EN = {
    "Apple": "apple", "Blueberry": "blueberry", "Cherry_(including_sour)": "cherry",
    "Corn_(maize)": "corn", "Grape": "grape", "Orange": "orange", "Peach": "peach",
    "Pepper,_bell": "bell pepper", "Potato": "potato", "Raspberry": "raspberry",
    "Soybean": "soybean", "Squash": "squash", "Strawberry": "strawberry", "Tomato": "tomato",
}
ENFERMEDAD_EN = {
    "Apple_scab": "apple scab", "Black_rot": "black rot",
    "Cedar_apple_rust": "cedar apple rust", "Powdery_mildew": "powdery mildew",
    "Cercospora_leaf_spot Gray_leaf_spot": "gray leaf spot", "Common_rust_": "common rust",
    "Northern_Leaf_Blight": "northern leaf blight", "Esca_(Black_Measles)": "esca black measles",
    "Leaf_blight_(Isariopsis_Leaf_Spot)": "isariopsis leaf spot",
    "Haunglongbing_(Citrus_greening)": "citrus greening huanglongbing",
    "Bacterial_spot": "bacterial spot", "Early_blight": "early blight",
    "Late_blight": "late blight", "Leaf_Mold": "leaf mold",
    "Septoria_leaf_spot": "septoria leaf spot",
    "Spider_mites Two-spotted_spider_mite": "two spotted spider mite damage",
    "Target_Spot": "target spot",
    "Tomato_Yellow_Leaf_Curl_Virus": "yellow leaf curl virus",
    "Tomato_mosaic_virus": "mosaic virus", "Leaf_scorch": "leaf scorch",
}
PLANTILLAS = {
    "P1_simple": (lambda c, e: f"a photo of a {c} leaf with {e}",
                  lambda c: f"a photo of a healthy {c} leaf"),
    "P2_clinica": (lambda c, e: f"a close-up photograph of a {c} leaf showing symptoms of {e}, plant disease",
                   lambda c: f"a close-up photograph of a healthy {c} leaf, no disease symptoms"),
    "P3_escueta": (lambda c, e: f"{e} on {c}", lambda c: f"healthy {c}"),
}
PROMPTS_HOJA = ["a photo of a plant leaf", "a close-up photo of green foliage",
                "a photo of a crop plant"]
PROMPTS_NO_HOJA = ["a photo of a person", "a photo of an animal",
                   "a photo of a document with text", "a photo of a vehicle",
                   "a photo of a building", "a photo of food on a plate",
                   "a screenshot of a user interface"]

_estado = {}


def _partes(label):
    cultivo, _, enfermedad = label.partition("___")
    return CULTIVO_EN[cultivo], (None if enfermedad == "healthy" else ENFERMEDAD_EN[enfermedad])


def textos_de_clases(plantilla, etiquetas=None):
    f_enf, f_sano = PLANTILLAS[plantilla]
    out = []
    for label in (etiquetas or etiquetas_pv()):
        c, e = _partes(label)
        out.append(f_sano(c) if e is None else f_enf(c, e))
    return out


def emb_textos(textos, lote=64):
    import torch
    from .modelos import clip
    modelo, proc = clip()
    salida = []
    with torch.no_grad():
        for i in range(0, len(textos), lote):
            ent = proc(text=textos[i:i + lote], return_tensors="pt",
                       padding=True, truncation=True)
            v = modelo.get_text_features(**ent)
            v = v if isinstance(v, torch.Tensor) else v.pooler_output
            salida.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    return np.concatenate(salida, axis=0).astype(np.float32)


def emb_imagenes(imagenes, lote=32):
    import torch
    from .modelos import clip
    modelo, proc = clip()
    salida = []
    with torch.no_grad():
        for i in range(0, len(imagenes), lote):
            ent = proc(images=imagenes[i:i + lote], return_tensors="pt")
            v = modelo.get_image_features(**ent)
            v = v if isinstance(v, torch.Tensor) else v.pooler_output
            salida.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    return np.concatenate(salida, axis=0).astype(np.float32)


def cargar():
    """Carga los artefactos de M4. Los embeddings de texto se leen precalculados si existen
    (así el arranque no pasa 45 textos por el encoder de CLIP)."""
    if _estado:
        return _estado
    import joblib

    from .modelos import clip

    with open(DIR_VISION / "vision_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    cal = cfg["calibracion"]

    etiquetas = cfg["etiquetas"]
    assert etiquetas == etiquetas_pv(), "las 38 clases de vision/ no son las del catálogo"

    modelo, _ = clip()
    escala = float(modelo.logit_scale.exp().detach().cpu())

    p_txt, p_hoja = DIR_VISION / "texto_clases.npy", DIR_VISION / "texto_hoja.npy"
    texto_clases = (np.load(p_txt) if p_txt.exists()
                    else emb_textos(textos_de_clases(cfg["plantilla"], etiquetas)))
    v_hoja = (np.load(p_hoja) if p_hoja.exists()
              else emb_textos(PROMPTS_HOJA + PROMPTS_NO_HOJA))

    # Punto de operación. Los umbrales del artefacto son los CALIBRADOS en M4 (percentil 5
    # del retenido, que es lo que declara la tasa de falsas alarmas). Se pueden mover en
    # producción con una variable de entorno -- por ejemplo, bajar `umbral_proto` hace que
    # la válvula dispare menos y deje pasar fotos de campo más lejanas a lo entrenado.
    # Se hace así, y no editando el artefacto, para que el calibrado siga siendo
    # reproducible y para que el health check pueda decir con qué se está operando.
    def _umbral(nombre, calibrado):
        bruto = os.environ.get(f"AGRO_UMBRAL_{nombre.upper()}")
        try:
            return float(bruto) if bruto else float(calibrado)
        except ValueError:
            return float(calibrado)

    operativo = {k: _umbral(k, cal[f"umbral_{k}"]) for k in ("p", "proto", "hoja")}
    calibrado = {k: float(cal[f"umbral_{k}"]) for k in ("p", "proto", "hoja")}

    _estado.update({
        "cfg": cfg, "modo": cfg["modo"], "plantilla": cfg["plantilla"], "etiquetas": etiquetas,
        "umbral_p": operativo["p"], "umbral_proto": operativo["proto"],
        "umbral_hoja": operativo["hoja"], "escala": escala,
        "umbrales_calibrados": calibrado, "umbrales_operativos": operativo,
        "umbrales_movidos": {k: (calibrado[k], operativo[k])
                             for k in operativo if operativo[k] != calibrado[k]},
        "texto_clases": texto_clases.astype(np.float32),
        "v_hoja": v_hoja.astype(np.float32), "n_hoja": len(PROMPTS_HOJA),
        "prototipos": np.load(DIR_VISION / "prototipos.npy"),
        "probe": joblib.load(DIR_VISION / "clasificador_m4.joblib"),
    })
    return _estado


def _p_es_hoja(v_img):
    e = _estado
    logits = e["escala"] * (v_img @ e["v_hoja"].T)
    p = np.exp(logits - logits.max())
    return float((p / p.sum())[:e["n_hoja"]].sum())


def ver(imagen):
    """imagen (PIL) -> dict con la etiqueta propuesta y el estado de las guardas.

    NO decide sola: devuelve 'valvula' y el motivo. Quien decide qué hacer con eso es
    `agro.sistema.responder`. Separar medir de actuar es lo que permitió medir la válvula
    en M4 y lo que permite registrarla en los logs de M5.
    """
    e = cargar()
    t0 = time.time()
    v = emb_imagenes([imagen])[0]
    etiquetas = e["etiquetas"]

    if e["modo"] == "probe" and e["probe"] is not None:
        probs = e["probe"].predict_proba(v.reshape(1, -1))[0]
        clases = list(e["probe"].classes_)
        orden = np.argsort(-probs)
        top = [(clases[i], float(probs[i])) for i in orden[:3]]
    else:
        logits = e["escala"] * (v @ e["texto_clases"].T)
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        orden = np.argsort(-probs)
        top = [(etiquetas[i], float(probs[i])) for i in orden[:3]]

    etiqueta, p1 = top[0]
    p2 = top[1][1] if len(top) > 1 else 0.0
    i = etiquetas.index(etiqueta)
    cos_proto = float(v @ e["prototipos"][i])
    p_hoja = _p_es_hoja(v)

    motivos = []
    if p_hoja < e["umbral_hoja"]:
        motivos.append(f"la imagen no parece una hoja de planta (P={p_hoja:.2f})")
    if p1 < e["umbral_p"]:
        motivos.append(f"ninguna de las 38 clases alcanza confianza suficiente (max={p1:.2f})")
    if cos_proto < e["umbral_proto"]:
        motivos.append("la imagen está lejos de todo lo que el clasificador vio entrenando "
                       f"(coseno={cos_proto:.2f})")

    return {"etiqueta": etiqueta, "p_top1": round(p1, 4), "margen": round(p1 - p2, 4),
            "cos_prototipo": round(cos_proto, 4), "p_hoja": round(p_hoja, 4),
            "es_hoja": p_hoja >= e["umbral_hoja"],
            "top3": [(et, round(p, 4)) for et, p in top],
            "valvula": bool(motivos), "motivos": motivos,
            "latencia_s": round(time.time() - t0, 3), "modo": e["modo"]}
