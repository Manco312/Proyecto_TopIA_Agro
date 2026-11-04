"""Recuperación: densa + BM25 + RRF + reranking. Las mismas funciones de M3 §8-§10.

Diferencia de M5: el índice **no se construye al arrancar**. Los chunks y sus embeddings
viajan precalculados en `artefactos/indice/`; aquí sólo se cargan. En M3 el arranque
costaba leer 50 PDF, trocearlos y codificar ~500 chunks; aquí es un `np.load`.

Motor denso: producto punto sobre embeddings L2-normalizados. Es **el mismo ranking** que
daba Chroma con `hnsw:space=cosine` -- scripts/construir_indice.py lo comprueba caso por
caso contra Chroma y falla si difiere. A cambio, producción no arrastra chromadb.
"""
from __future__ import annotations

import json
import re

import numpy as np

from .config import DIR_INDICE, K_CANDIDATOS, K_FINAL, RRF_K
from .texto import _norm

_estado = {}


def cargar():
    if _estado:
        return _estado
    with open(DIR_INDICE / "chunks.json", encoding="utf-8") as f:
        d = json.load(f)
    emb = np.load(DIR_INDICE / "embeddings.npy")
    assert len(d["chunks"]) == emb.shape[0], "el índice y sus embeddings no cuadran"
    _estado.update({"chunks": d["chunks"], "metadatos": d["metadatos"], "ids": d["ids"],
                    "emb": emb.astype(np.float32), "bm25": _bm25(d["chunks"])})
    return _estado


def _bm25(chunks):
    from rank_bm25 import BM25Okapi
    return BM25Okapi([tok_bm25(c) for c in chunks])


def tok_bm25(texto):
    """Minúsculas, sin tildes y sin puntuación ("Envés," == "enves"). M3 §9, v2."""
    return re.findall(r"[a-z0-9]+", _norm(texto))


def chunks():
    return cargar()["chunks"]


def metadatos():
    return cargar()["metadatos"]


def ids():
    return cargar()["ids"]


def buscar_densa(consulta, k=K_CANDIDATOS):
    from .modelos import embebedor
    e = cargar()
    q = embebedor().encode([consulta], normalize_embeddings=True)[0].astype(np.float32)
    sims = e["emb"] @ q
    k = min(k, len(sims))
    top = np.argpartition(-sims, k - 1)[:k]
    return [int(i) for i in top[np.argsort(-sims[top], kind="stable")]]


def buscar_bm25(consulta, k=K_CANDIDATOS):
    e = cargar()
    scores = e["bm25"].get_scores(tok_bm25(consulta))
    return [int(i) for i in np.argsort(scores)[::-1][:k]]


def buscar_hibrida(consulta, k=K_CANDIDATOS, krrf=RRF_K):
    listas = [buscar_densa(consulta, k), buscar_bm25(consulta, k)]
    puntos = {}
    for lista in listas:
        for puesto, idx in enumerate(lista):
            puntos[idx] = puntos.get(idx, 0) + 1.0 / (krrf + puesto + 1)
    return [idx for idx, _ in sorted(puntos.items(), key=lambda x: -x[1])][:k]


def buscar_con_rerank(consulta, k_recuperar=K_CANDIDATOS, k_final=K_FINAL):
    from .modelos import reranker
    candidatos = buscar_hibrida(consulta, k_recuperar)
    if not candidatos:
        return []
    cs = chunks()
    scores = reranker().predict([(consulta, cs[i]) for i in candidatos])
    orden = np.argsort(scores)[::-1]
    return [candidatos[i] for i in orden[:k_final]]
