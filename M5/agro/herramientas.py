"""Las tres herramientas de M3 §13 y la redacción final. Lógica sin cambios.

  1. `diagnostico_diferencial(cultivo, sintomas)` -- determinista, sin LLM.
  2. `consultar_ficha(etiqueta)`  <- la interfaz que usa el clasificador de M4.
  3. `buscar_en_fichas(consulta)` -- el RAG avanzado de M3 §10.

`_redactar_final` es el único punto donde el sistema llama al redactor.
"""
from __future__ import annotations

import difflib

import numpy as np

from . import indice
from .catalogo import (cat_keys, cat_textos, catalogo, chunks_por_doc, cultivo_vacio,
                       cultivos_cubiertos, detectar_cultivos)
from .config import DIAG_RANKER, K_CANDIDATOS, K_FINAL, MAX_CTX_CHARS, MAX_NEW_SISTEMA
from .texto import _norm

SYSTEM_RAG = (
    "Instruccion: Responde SOLO con base en el CONTEXTO proporcionado. "
    "Si la respuesta no esta en el contexto, di claramente: "
    '"No tengo esa informacion en mis fuentes." No inventes datos. '
    "Cita la fuente del contexto que uses (institucion o guia). Se breve y claro. "
    "Usa el formato: Se trata de <problema> (<agente causal>). Identificacion: ... "
    "Accion recomendada: ... Prevencion a futuro: ...")


# --------------------------------------------------------------------------------------
# Herramienta 1 -- diagnóstico diferencial
# --------------------------------------------------------------------------------------
def _candidato(i, puntaje):
    c = catalogo()[cat_keys()[i]]
    return {"etiqueta": c["etiqueta"], "cultivo": c["cultivo"], "problema": c["problema"],
            "patogeno": c["patogeno"], "tipo": c["tipo"], "puntaje": round(float(puntaje), 3),
            "en_plantvillage": c["en_plantvillage"],
            "fuentes": ["Colombia" if d.startswith("ica_") else "extension agricola"
                        for d in c["doc_ids"]]}


def _rankear(idxs, texto):
    """Ordena entradas del catálogo por afinidad con los síntomas (cross-encoder, v2 de M3)."""
    from .modelos import embebedor, reranker
    idxs = list(idxs)
    textos = cat_textos()
    if DIAG_RANKER == "cross":
        s = np.asarray(reranker().predict([(texto, textos[i]) for i in idxs]), dtype=float)
    else:
        emb = embebedor().encode([textos[i] for i in idxs], normalize_embeddings=True)
        s = emb @ embebedor().encode([texto], normalize_embeddings=True)[0]
    return [(idxs[j], float(s[j])) for j in np.argsort(-s)]


def diagnostico_diferencial(cultivo, sintomas, top=3):
    """Determinista (sin LLM). estado = 'cubierto' | 'no_cubierto' | 'falta_cultivo'."""
    cat, keys = catalogo(), cat_keys()
    arg = _norm(cultivo or "").strip(" .,:;\"'")
    arg_vacio = cultivo_vacio(arg)
    cultivos = detectar_cultivos(arg) or detectar_cultivos(sintomas)
    texto = f"{cultivo or ''} {sintomas or ''}".strip()

    if not cultivos:
        if arg and not arg_vacio:
            return {"estado": "no_cubierto", "cultivo": cultivo.strip(), "candidatos": []}
        vistos, cands = set(), []
        enfermos = [i for i, k in enumerate(keys) if cat[k]["tipo"] != "sano"]
        for i, p in _rankear(enfermos, texto):
            ck = cat[keys[i]]["cultivo_key"]
            if ck in vistos:
                continue
            vistos.add(ck)
            cands.append(_candidato(i, p))
            if len(cands) == 4:
                break
        return {"estado": "falta_cultivo", "cultivo": None, "candidatos": cands}

    key = cultivos[0]
    del_cultivo = [i for i, k in enumerate(keys) if cat[k]["cultivo_key"] == key]
    return {"estado": "cubierto", "cultivo": key,
            "candidatos": [_candidato(i, p) for i, p in _rankear(del_cultivo, texto)[:top]]}


def _obs_diagnostico(d):
    if d["estado"] == "no_cubierto":
        return (f"El cultivo '{d['cultivo']}' NO esta cubierto por las fichas del sistema. "
                "Escribe Responder[ok]: el sistema se abstendra.")
    if d["estado"] == "falta_cultivo":
        pos = "; ".join(f"{c['problema']} ({c['cultivo']})" for c in d["candidatos"])
        return (f"La pregunta no dice el cultivo. Posibles, segun el sintoma: {pos}. "
                "Escribe Responder[ok]: el sistema pedira aclarar el cultivo.")
    cands = "; ".join(f"{c['etiqueta']} = {c['problema']} ({c['patogeno'] or 'sano'}, {c['tipo']}) "
                      f"[puntaje {c['puntaje']}]" for c in d["candidatos"])
    return f"Cultivo cubierto: {d['cultivo']}. Candidatos: {cands}"


# --------------------------------------------------------------------------------------
# Herramienta 2 -- consultar la ficha de una etiqueta (la que usa el clasificador de M4)
# --------------------------------------------------------------------------------------
def _resolver_etiqueta(etiqueta):
    cat = catalogo()
    if etiqueta in cat:
        return etiqueta
    por_norm = {_norm(k): k for k in cat}
    e = _norm(etiqueta or "").strip(" '\"")
    if e in por_norm:
        return por_norm[e]
    m = difflib.get_close_matches(e, list(por_norm), n=1, cutoff=0.6)
    return por_norm[m[0]] if m else None


def consultar_ficha(etiqueta):
    """(etiqueta_resuelta, índices de chunks). Ficha de Colombia primero (más específica)."""
    k = _resolver_etiqueta(etiqueta)
    if k is None:
        return None, []
    porc = chunks_por_doc()
    docs = sorted(catalogo()[k]["doc_ids"], key=lambda d: not d.startswith("ica_"))
    return k, [i for d in docs for i in porc.get(d, [])]


# --------------------------------------------------------------------------------------
# Herramienta 3 -- búsqueda libre
# --------------------------------------------------------------------------------------
def buscar_en_fichas(consulta):
    return indice.buscar_con_rerank(consulta, K_CANDIDATOS, K_FINAL)


# --------------------------------------------------------------------------------------
# Redacción final
# --------------------------------------------------------------------------------------
def _respuesta_abstencion(d):
    return (f"No tengo fichas tecnicas para el cultivo '{d['cultivo']}': esta fuera de las 38 clases de "
            "PlantVillage y de las fichas de Colombia que consulta este sistema, asi que no puedo darte una "
            "recomendacion sin inventarla. Consulta a un agronomo o a la oficina del ICA mas cercana. "
            f"Cultivos que si cubro: {', '.join(cultivos_cubiertos())}.")


def _respuesta_aclaracion(d):
    pos = "; ".join(f"{c['problema']} en {c['cultivo']}" for c in d["candidatos"])
    cults = ", ".join(dict.fromkeys(c["cultivo"] for c in d["candidatos"]))
    return ("No indicaste de que cultivo se trata, y ese sintoma coincide con problemas de cultivos distintos "
            f"que se manejan de forma distinta: {pos}. Antes de recomendarte algo, dime que cultivo es "
            f"(por ejemplo: {cults}) y te doy el manejo concreto segun la ficha.")


def _dedup_cap(idxs, max_chars=MAX_CTX_CHARS):
    cs = indice.chunks()
    out, vistos, total = [], set(), 0
    for i in idxs:
        if i in vistos:
            continue
        if out and total + len(cs[i]) > max_chars:
            break
        out.append(i)
        vistos.add(i)
        total += len(cs[i])
    return out


def armar_contexto(idxs_chunks):
    cs, ms = indice.chunks(), indice.metadatos()
    return "\n\n".join(f"[Fuente: {ms[i]['fuente'][:90]}]\n{cs[i]}" for i in idxs_chunks)


def generar_rag(pregunta, idxs_chunks, redactor, max_new=None):
    prompt = (f"{SYSTEM_RAG}\n\nContexto:\n{armar_contexto(idxs_chunks)}"
              f"\n\nPregunta: {pregunta}\nRespuesta:")
    return redactor.generar(prompt, max_new)


def sistema_rag_avanzado(pregunta, redactor, k=K_FINAL):
    return generar_rag(pregunta, indice.buscar_con_rerank(pregunta, K_CANDIDATOS, k), redactor)


def redactar_final(pregunta, estado, redactor):
    """Devuelve (texto, traza). La traza es lo que se registra en el log por petición."""
    cat = catalogo()
    d = estado.get("diag")
    if d is None:                                    # precondición barata y determinista
        d = estado["diag"] = diagnostico_diferencial("", pregunta)

    if d["estado"] == "no_cubierto":
        return _respuesta_abstencion(d), {"salida": "abstencion", "etiqueta": None, "chunks": []}
    if d["estado"] == "falta_cultivo":
        return _respuesta_aclaracion(d), {"salida": "aclaracion", "etiqueta": None, "chunks": []}

    if not estado["idxs"] and d["candidatos"]:       # el planificador no pidió ficha
        estado["etiqueta"], estado["idxs"] = consultar_ficha(d["candidatos"][0]["etiqueta"])
    idxs = _dedup_cap(estado["idxs"])
    if not idxs:
        return sistema_rag_avanzado(pregunta, redactor), {"salida": "rag_avanzado",
                                                          "etiqueta": None, "chunks": []}

    c = cat.get(estado.get("etiqueta") or "") or cat[d["candidatos"][0]["etiqueta"]]
    nota = (f"Diagnostico segun el catalogo: {c['problema']}"
            + (f" ({c['patogeno']}, {c['tipo']})." if c["patogeno"] else " (planta sana, sin patogeno)."))
    prompt = (f"{SYSTEM_RAG}\n\nContexto:\n{nota}\n\n{armar_contexto(idxs)}"
              f"\n\nPregunta: {pregunta}\nRespuesta:")
    ms = indice.metadatos()
    return redactor.generar(prompt), {"salida": "ficha", "etiqueta": c["etiqueta"],
                                      "chunks": [ms[i]["doc_id"] for i in idxs]}


def ejecutar_herramienta(nombre, args, pregunta, estado):
    """Despacha una llamada del planificador y actualiza el estado."""
    if nombre == "diagnostico_diferencial":
        estado["diag"] = diagnostico_diferencial(
            args.get("cultivo", ""), f"{args.get('sintomas', '')} {pregunta}".strip())
        return _obs_diagnostico(estado["diag"])
    if nombre == "consultar_ficha":
        k, idxs = consultar_ficha(args.get("etiqueta", ""))
        if k is None:
            return "Etiqueta no encontrada en el catalogo. Usa una etiqueta de la lista del diagnostico."
        estado["etiqueta"] = k
        estado["idxs"].extend(idxs)
        return f"Ficha {k} recuperada ({len(idxs)} pasajes): {catalogo()[k]['problema']}."
    if nombre == "buscar_en_fichas":
        idxs = buscar_en_fichas(args.get("consulta", pregunta))
        estado["idxs"].extend(idxs)
        return "Pasajes de: " + ", ".join(indice.metadatos()[i]["doc_id"] for i in idxs)
    return f"error: herramienta desconocida '{nombre}'"


def diag_desde_etiqueta(etiqueta):
    """El 'diag' que espera `redactar_final`, pero con la etiqueta que vino de la FOTO en vez
    de la que el diagnóstico diferencial adivinaría desde el texto (M4 §5)."""
    k, _ = consultar_ficha(etiqueta)
    if k is None:
        return None
    i = cat_keys().index(k)
    return {"estado": "cubierto", "cultivo": catalogo()[k]["cultivo_key"],
            "candidatos": [_candidato(i, 1.0)]}
