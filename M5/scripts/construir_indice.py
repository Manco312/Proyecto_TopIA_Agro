"""Precalcula el índice del RAG y los artefactos de visión que viajan al Space.

Se corre **una vez, fuera de producción**, y deja en `artefactos/` todo lo que el Space
necesita para arrancar sin leer un solo PDF. El cuidado de despliegue de la S15 era
explícito: *no construyan el índice al arrancar; súbanlo ya construido*.

Además de construir, **verifica**. Cuatro cosas, y si alguna falla el script aborta en vez
de dejar un artefacto silenciosamente distinto al de M3:

  1. el troceo da los 151 chunks de M3 §7;
  2. la búsqueda densa con numpy devuelve el MISMO ranking que daba Chroma (`--chroma`);
  3. las tres consultas con salida registrada en notebookM3 dan lo mismo;
  4. ninguna de las 50 fichas contiene una dosis numérica -- que es la premisa sobre la
     que se apoya la regla 3 del filtro de salida de M5.

Uso:
    python scripts/construir_indice.py                 # construye y verifica
    python scripts/construir_indice.py --chroma        # además compara contra chromadb
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agro import config                                                    # noqa: E402
from agro.catalogo import construir_catalogo, textos_de_catalogo           # noqa: E402
from agro.texto import DOSIS_NUMERICA, _norm                               # noqa: E402

CHUNKS_ESPERADOS = 151          # M3 §7, con CHUNKING="seccion"
DOCS_ESPERADOS = 50             # 38 de entrenamiento + 12 de fuente externa

# Salidas registradas en notebookM3.ipynb: si el índice reconstruido no las reproduce,
# el sistema desplegado no es el que se evaluó.
REGRESION = [
    ("densa", "mi papa tiene manchas oscuras y moho blanco", 3,
     ["base_Grape___healthy_c0", "base_Potato___healthy_c0",
      "base_Corn_(maize)___Northern_Leaf_Blight_c0"]),
    ("densa", "que recomienda el ICA para el HLB en citricos", 3,
     ["ica_ica-06-citricos-hlb_c0", "ica_ica-06-citricos-hlb_c2",
      "ica_ica-04-vid-tizon-foliar_c1"]),
    ("bm25", "que recomienda el ICA para el HLB en citricos", 3,
     ["ica_ica-06-citricos-hlb_c0", "ica_ica-06-citricos-hlb_c2",
      "ica_ica-06-citricos-hlb_c1"]),
    ("hibrida", "que recomienda el ICA para el HLB en citricos", 3,
     ["ica_ica-06-citricos-hlb_c0", "ica_ica-06-citricos-hlb_c2",
      "ica_ica-04-vid-tizon-foliar_c1"]),
    ("hibrida", "polvillo naranja rojizo en el enves de la hoja", 4,
     ["base_Tomato___Leaf_Mold_c0", "base_Strawberry___Leaf_scorch_c0",
      "ica_ica-11-cafe-roya_c0", "base_Corn_(maize)___Common_rust__c0"]),
    ("rerank", "polvillo naranja rojizo en el enves de la hoja", 4,
     ["ica_ica-11-cafe-roya_c0", "base_Apple___Cedar_apple_rust_c0",
      "base_Tomato___Spider_mites Two-spotted_spider_mite_c0", "base_Potato___Late_blight_c0"]),
]

# --------------------------------------------------------------------------------------
# Troceo -- M3 §7 copiado sin cambios
# --------------------------------------------------------------------------------------
_NOMBRE_SECCION = {"identificacion": "Identificación",
                   "manejo recomendado": "Acción recomendada",
                   "prevencion": "Prevención a futuro"}
_FIN_FICHA = ("Fuente:", "Referencia:", "Ficha elaborada por el equipo")


def _slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def partir_en_chunks(texto, size=config.CHUNK_SIZE, overlap=config.OVERLAP):
    trozos, inicio = [], 0
    while inicio < len(texto):
        trozos.append(texto[inicio:inicio + size])
        inicio += size - overlap
    return trozos


def parsear_ficha(texto):
    lineas = [l.strip() for l in texto.splitlines() if l.strip()]
    i_meta = next((i for i, l in enumerate(lineas) if l.startswith("Cultivo:")), None)
    if i_meta is None or i_meta < 1:
        return None
    institucion, titulo = lineas[0], " ".join(lineas[1:i_meta])
    meta, secciones, actual = [], [], None
    for l in lineas[i_meta:]:
        if l.startswith(_FIN_FICHA):
            break
        clave = _norm(l)
        if clave in _NOMBRE_SECCION:
            actual = [_NOMBRE_SECCION[clave], []]
            secciones.append(actual)
        elif actual is None:
            meta.append(l)
        else:
            actual[1].append(l)
    meta = " · ".join(p.strip() for p in " ".join(meta).split("·")
                           if p.strip().startswith(("Cultivo", "Agente causal")))
    secciones = [(nombre, " ".join(ls)) for nombre, ls in secciones if ls]
    if not secciones:
        return None
    return {"institucion": institucion, "titulo": titulo, "meta": meta, "secciones": secciones}


def _partir_por_oraciones(texto, size=config.CHUNK_SIZE):
    piezas, actual = [], ""
    for o in (o for o in re.split(r"(?<=[.;])\s+", texto) if o):
        if actual and len(actual) + 1 + len(o) > size:
            piezas.append(actual)
            actual = o
        else:
            actual = f"{actual} {o}".strip()
    if actual:
        piezas.append(actual)
    return [q for p in piezas for q in (partir_en_chunks(p) if len(p) > size else [p])]


def chunks_por_seccion(doc):
    ficha = parsear_ficha(doc["texto"])
    if ficha is None:
        return partir_en_chunks(doc["texto"])
    origen = ("Ficha" if doc["fuente_tipo"] == "entrenamiento"
              else f"Ficha {ficha['institucion']} (Colombia)")
    cabecera = f"{origen}: {ficha['titulo']} | {ficha['meta']}"
    return [f"{cabecera} | {nombre}: {pieza}"
            for nombre, cuerpo in ficha["secciones"] for pieza in _partir_por_oraciones(cuerpo)]


# --------------------------------------------------------------------------------------
def leer_corpus(origen: Path):
    import pypdf

    with open(origen / "datos" / "base_conocimiento_plantvillage.json", encoding="utf-8") as f:
        base = json.load(f)
    with open(origen / "datos" / "corpus_ica_colombia.json", encoding="utf-8") as f:
        externo = json.load(f)

    d_train = origen / "datos" / "pdfs" / "entrenamiento"
    d_ext = origen / "datos" / "pdfs" / "externo"

    def texto_pdf(p):
        return "\n".join(pg.extract_text() or "" for pg in pypdf.PdfReader(str(p)).pages).strip()

    docs = []
    for label, r in base.items():
        p = d_train / (_slug(label) + ".pdf")
        docs.append({"id": f"base_{label}", "fuente": r["fuente"], "fuente_tipo": "entrenamiento",
                     "institucion": None, "archivo_pdf": p.name, "texto": texto_pdf(p)})
    for r in externo:
        p = d_ext / (_slug(r["id"]) + ".pdf")
        docs.append({"id": f"ica_{r['id']}", "fuente": r["fuente"],
                     "fuente_tipo": r.get("fuente_tipo", "externa_no_entrenamiento"),
                     "institucion": r.get("institucion"), "archivo_pdf": p.name,
                     "texto": texto_pdf(p)})
    return base, externo, docs


def verificar_sin_dosis(base, externo):
    """La premisa de la regla 3 del filtro de salida. Si algún día una ficha trae una dosis,
    el filtro dejaría de ser correcto y este script tiene que avisar."""
    fallos = []
    for nombre, registros in (("base", base.values()), ("ica", externo)):
        for r in registros:
            for campo in ("identificacion", "tratamiento", "prevencion"):
                m = DOSIS_NUMERICA.search(_norm(r.get(campo, "")))
                if m:
                    fallos.append(f"{nombre}:{r.get('problema')}:{campo} -> {m.group(0)!r}")
    return fallos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", default=str(config.ORIGEN_DATOS))
    ap.add_argument("--chroma", action="store_true",
                    help="comparar el ranking denso de numpy contra chromadb")
    args = ap.parse_args()

    origen = Path(args.origen)
    assert (origen / "datos").is_dir(), f"no encuentro {origen}/datos"
    t0 = time.time()

    print(f"1. Leyendo el corpus de {origen} ...")
    base, externo, docs = leer_corpus(origen)
    assert len(docs) == DOCS_ESPERADOS, f"{len(docs)} documentos, esperaba {DOCS_ESPERADOS}"
    print(f"   {len(docs)} documentos PDF "
          f"({sum(d['fuente_tipo'] == 'entrenamiento' for d in docs)} entrenamiento + "
          f"{sum(d['fuente_tipo'] != 'entrenamiento' for d in docs)} externos)")

    print("2. Verificando la premisa del filtro de salida (ninguna ficha trae dosis) ...")
    fallos = verificar_sin_dosis(base, externo)
    assert not fallos, ("hay dosis numéricas en el corpus; la regla 3 del filtro de salida "
                        "dejaría de ser correcta:\n  " + "\n  ".join(fallos))
    print("   OK: 0 dosis numéricas en las 50 fichas.")

    print("3. Troceando (CHUNKING='seccion') ...")
    chunks, metadatos, ids = [], [], []
    for doc in docs:
        assert parsear_ficha(doc["texto"]) is not None, f"ficha sin estructura: {doc['id']}"
        for j, ch in enumerate(chunks_por_seccion(doc)):
            chunks.append(ch)
            metadatos.append({"fuente": doc["fuente"], "fuente_tipo": doc["fuente_tipo"],
                              "doc_id": doc["id"], "institucion": doc["institucion"] or "",
                              "archivo_pdf": doc["archivo_pdf"]})
            ids.append(f"{doc['id']}_c{j}")
    assert len(chunks) == CHUNKS_ESPERADOS, \
        f"{len(chunks)} chunks, M3 produjo {CHUNKS_ESPERADOS}: el troceo cambió"
    print(f"   {len(chunks)} chunks (igual que M3 §7)")

    print("4. Codificando con", config.EMBED_ID, "...")
    from agro.modelos import embebedor
    emb = embebedor().encode(chunks, show_progress_bar=False,
                             normalize_embeddings=True).astype(np.float32)

    print("5. Catálogo y embeddings del catálogo ...")
    catalogo = construir_catalogo(base, externo)
    cat_textos = textos_de_catalogo(catalogo)
    chunks_por_doc = {}
    for i, m in enumerate(metadatos):
        chunks_por_doc.setdefault(m["doc_id"], []).append(i)
    print(f"   {len(catalogo)} etiquetas "
          f"({sum(c['en_plantvillage'] for c in catalogo.values())} de PlantVillage)")

    config.DIR_INDICE.mkdir(parents=True, exist_ok=True)
    with open(config.DIR_INDICE / "chunks.json", "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "metadatos": metadatos, "ids": ids}, f, ensure_ascii=False)
    np.save(config.DIR_INDICE / "embeddings.npy", emb)
    with open(config.DIR_INDICE / "catalogo.json", "w", encoding="utf-8") as f:
        json.dump({"catalogo": catalogo, "cat_textos": cat_textos,
                   "chunks_por_doc": chunks_por_doc}, f, ensure_ascii=False)

    print("6. Embeddings de texto de CLIP (para que el Space no los calcule al arrancar) ...")
    from agro import vision
    with open(config.DIR_VISION / "vision_config.json", encoding="utf-8") as f:
        vcfg = json.load(f)
    np.save(config.DIR_VISION / "texto_clases.npy",
            vision.emb_textos(vision.textos_de_clases(vcfg["plantilla"], vcfg["etiquetas"])))
    np.save(config.DIR_VISION / "texto_hoja.npy",
            vision.emb_textos(vision.PROMPTS_HOJA + vision.PROMPTS_NO_HOJA))

    print("7. Regresión del retrieval contra las salidas registradas en notebookM3 ...")
    from agro import indice
    indice._estado.clear()
    fallos = []
    for modo, consulta, k, esperado in REGRESION:
        fn = {"densa": indice.buscar_densa, "bm25": indice.buscar_bm25,
              "hibrida": indice.buscar_hibrida}.get(modo)
        obtenido = ([ids[i] for i in indice.buscar_con_rerank(consulta, config.K_CANDIDATOS, k)]
                    if modo == "rerank" else [ids[i] for i in fn(consulta, k)])
        ok = obtenido == esperado
        print(f"   [{'OK ' if ok else 'DIF'}] {modo:8s} {consulta[:46]!r}")
        if not ok:
            fallos.append(f"{modo} {consulta!r}\n      esperado: {esperado}\n      obtenido: {obtenido}")
    if fallos:
        print("\n   AVISO: el retrieval no reproduce M3:\n      " + "\n      ".join(fallos))

    paridad = None
    if args.chroma:
        print("8. Comparando numpy contra chromadb ...")
        import chromadb
        cli = chromadb.Client()
        try:
            cli.delete_collection("corpus_m5")
        except Exception:
            pass
        col = cli.create_collection("corpus_m5", metadata={"hnsw:space": "cosine"})
        col.add(ids=ids, documents=chunks, metadatas=metadatos, embeddings=emb.tolist())
        iguales, total = 0, 0
        for _, consulta, _, _ in REGRESION:
            q = embebedor().encode([consulta], normalize_embeddings=True)
            r = col.query(query_embeddings=q.tolist(), n_results=10)
            total += 1
            iguales += int([ids.index(i) for i in r["ids"][0]] == indice.buscar_densa(consulta, 10))
        paridad = f"{iguales}/{total}"
        print(f"   top-10 idéntico en {paridad} consultas")

    manifiesto = {
        "construido": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "version": config.VERSION,
        "origen": str(origen),
        "documentos": len(docs), "chunks": len(chunks), "etiquetas": len(catalogo),
        "chunking": {"modo": "seccion", "size": config.CHUNK_SIZE, "overlap": config.OVERLAP},
        "embed_id": config.EMBED_ID, "rerank_id": config.RERANK_ID, "clip_id": config.CLIP_ID,
        "dimension": int(emb.shape[1]),
        "regresion_retrieval": {"casos": len(REGRESION), "fallos": len(fallos)},
        "paridad_chroma": paridad,
        "dosis_en_corpus": 0,
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                   for p in sorted(config.DIR_INDICE.glob("*")) if p.is_file()},
        "entorno": {"python": platform.python_version(), "sistema": platform.system()},
    }
    for mod in ("torch", "transformers", "sentence_transformers", "sklearn", "numpy"):
        try:
            manifiesto["entorno"][mod] = __import__(mod).__version__
        except Exception:
            pass
    with open(config.DIR_INDICE / "manifiesto.json", "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, ensure_ascii=False, indent=1)

    mb = sum(os.path.getsize(p) for p in config.DIR_INDICE.glob("*")) / 2**20
    print(f"\nListo en {time.time() - t0:.0f} s. artefactos/indice/ = {mb:.1f} MB")
    print(json.dumps({k: manifiesto[k] for k in
                      ("documentos", "chunks", "etiquetas", "regresion_retrieval",
                       "paridad_chroma")}, ensure_ascii=False))
    return 0 if not fallos else 1


if __name__ == "__main__":
    raise SystemExit(main())
