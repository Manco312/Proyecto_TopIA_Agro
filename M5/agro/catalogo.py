"""El catálogo de etiquetas: el contrato entre el clasificador de imágenes (M4) y el
recomendador (M1-M3). Copiado de M3 §13, pero **leído de un artefacto precalculado** en vez
de reconstruirse al arrancar leyendo 50 PDF.
"""
from __future__ import annotations

import json
import re

from .config import DIR_INDICE
from .texto import _norm

# Nombres con que un agricultor se refiere a cada cultivo (texto normalizado, sin tildes).
# OJO: sin "naranja" -- también es un color ("polvillo naranja") y dispararía cultivos falsos.
ALIAS_CULTIVO = {
    "manzano": ["manzano", "manzanos", "manzana", "manzanas", "manzanal"],
    "arandano": ["arandano", "arandanos"],
    "cerezo": ["cerezo", "cerezos", "cereza", "cerezas"],
    "maiz": ["maiz", "maizal", "maizales", "choclo", "milpa"],
    "vid": ["vid", "vides", "uva", "uvas", "vinedo", "vinedos", "parra", "parras"],
    "naranjo": ["naranjo", "naranjos", "citrico", "citricos", "limonero", "limoneros", "mandarino"],
    "duraznero": ["duraznero", "durazneros", "durazno", "duraznos", "melocotonero", "melocoton"],
    "pimenton": ["pimenton", "pimentones", "pimiento", "pimientos"],
    "papa": ["papa", "papas", "patata", "patatas", "papero", "papera", "paperas"],
    "frambueso": ["frambueso", "frambuesos", "frambuesa", "frambuesas"],
    "soya": ["soya", "soja"],
    "calabaza": ["calabaza", "calabazas", "zapallo", "ahuyama", "auyama"],
    "fresa": ["fresa", "fresas", "frutilla", "frutillas"],
    "tomate": ["tomate", "tomates", "tomatera", "jitomate"],
    "cafe": ["cafe", "cafeto", "cafetos", "cafetal", "cafetales", "cafetero", "cafetera"],
    "cacao": ["cacao", "cacaotal", "cacaotales"],
}

_CULTIVO_VACIO = {"", "desconocido", "no especificado", "no indicado", "sin especificar", "ninguno",
                  "cultivo", "mi cultivo", "cultivos", "planta", "plantas", "hoja", "hojas", "arbol",
                  "arboles", "mata", "matas", "n/a", "na", "?", "no se", "indeterminado", "vacio",
                  "enves", "haz", "tallo", "tallos", "fruto", "frutos", "raiz", "raices", "rama", "ramas",
                  "flor", "flores", "brote", "brotes", "follaje", "lote", "lotes", "finca", "sintoma",
                  "sintomas", "polvillo", "polvo", "mancha", "manchas", "hongo", "plaga", "enfermedad"}


def cultivo_canonico(txt):
    return _norm(txt).split(" (")[0].strip()


def construir_catalogo(base_conocimiento, corpus_externo):
    """Build-time: el mismo bucle de M3 §13. Lo usa scripts/construir_indice.py."""
    catalogo = {}
    for label, r in base_conocimiento.items():
        catalogo[label] = {
            "etiqueta": label, "cultivo": r["cultivo"], "cultivo_key": cultivo_canonico(r["cultivo"]),
            "problema": r["problema"], "patogeno": r.get("patogeno"), "tipo": r["tipo"],
            "identificacion": r["identificacion"], "en_plantvillage": True,
            "doc_ids": [f"base_{label}"]}
    for r in corpus_externo:
        et = r["etiqueta"]
        if et in catalogo:
            catalogo[et]["doc_ids"].append(f"ica_{r['id']}")
        else:
            catalogo[et] = {
                "etiqueta": et, "cultivo": r["cultivo"], "cultivo_key": cultivo_canonico(r["cultivo"]),
                "problema": r["problema"], "patogeno": r.get("patogeno"), "tipo": r["tipo"],
                "identificacion": r["identificacion"], "en_plantvillage": False,
                "doc_ids": [f"ica_{r['id']}"]}
    sin_alias = {c["cultivo_key"] for c in catalogo.values()} - set(ALIAS_CULTIVO)
    assert not sin_alias, f"cultivos del catálogo sin alias: {sin_alias}"
    return catalogo


def textos_de_catalogo(catalogo):
    return [f"{c['problema']} en {c['cultivo']} ({c['patogeno'] or 'planta sana'}). "
            f"{c['identificacion']}" for c in catalogo.values()]


# --------------------------------------------------------------------------------------
# Runtime: se lee el artefacto. Sin pypdf, sin reportlab, sin 50 PDF.
# --------------------------------------------------------------------------------------
_estado = {}


def cargar():
    if _estado:
        return _estado
    with open(DIR_INDICE / "catalogo.json", encoding="utf-8") as f:
        d = json.load(f)
    catalogo = d["catalogo"]
    _estado.update({
        "catalogo": catalogo,
        "cat_keys": list(catalogo),
        "cat_textos": d["cat_textos"],
        "chunks_por_doc": {k: list(v) for k, v in d["chunks_por_doc"].items()},
        "cultivos_cubiertos": sorted({c["cultivo"] for c in catalogo.values()}),
        "etiquetas_pv": sorted(k for k, c in catalogo.items() if c["en_plantvillage"]),
    })
    return _estado


def catalogo():
    return cargar()["catalogo"]


def cat_keys():
    return cargar()["cat_keys"]


def cat_textos():
    return cargar()["cat_textos"]


def chunks_por_doc():
    return cargar()["chunks_por_doc"]


def cultivos_cubiertos():
    return cargar()["cultivos_cubiertos"]


def etiquetas_pv():
    return cargar()["etiquetas_pv"]


def detectar_cultivos(texto):
    n = _norm(texto or "")
    return [key for key, alias in ALIAS_CULTIVO.items()
            if any(re.search(rf"\b{re.escape(a)}\b", n) for a in alias)]


def cultivo_vacio(arg_norm):
    return (arg_norm in _CULTIVO_VACIO or arg_norm.startswith("no ")
            or len(arg_norm.split()) > 3)


# --------------------------------------------------------------------------------------
# Extractor de cultivo NO cubierto -- añadido en M5 por lo que destapó la ablación.
#
# En la ruta ReAct de M3 era el planificador de 1.5B quien sacaba "banano" de la pregunta y
# se lo pasaba a `diagnostico_diferencial`, que entonces respondía `no_cubierto` y el sistema
# se abstenía nombrando el límite. Sin planificador, `diagnostico_diferencial("", pregunta)`
# no encuentra ningún cultivo conocido y no puede distinguir "no dijo el cultivo" de "dijo
# uno que no cubro": acababa pidiendo aclaración en los dos casos. Eso degradaba la
# abstención, que es justo la propiedad de seguridad del sistema.
#
# Esto lo recupera sin LLM: patrones de posesivo/cultivo + un sustantivo, y sólo se consulta
# cuando `detectar_cultivos` ya devolvió vacío.
# --------------------------------------------------------------------------------------
_PATRONES_CULTIVO = [
    re.compile(r"\b(?:mis?|nuestr[oa]s?|el|la|los|las|un|una|unos|unas)\s+"
               r"(?:matas?|plantas?|cultivos?|arboles?|arbol|siembras?|palos?|lotes?|"
               r"huertas?|sembrados?)\s+de\s+([a-z]{3,})"),
    re.compile(r"\bcultivo\s+de\s+([a-z]{3,})"),
    re.compile(r"\b(?:mis?|nuestr[oa]s?)\s+([a-z]{3,})\b"),
]
_NO_ES_CULTIVO = _CULTIVO_VACIO | {
    "hoja", "hojas", "foto", "fotos", "imagen", "imagenes", "pregunta", "duda", "vecino",
    "abuelo", "papa_", "problema", "caso", "campo", "campos", "terreno", "parcela", "cosecha",
    "cosechas", "produccion", "semilla", "semillas", "riego", "suelo", "tierra", "agua",
    "amigo", "hijo", "familia", "region", "vereda", "municipio", "invernadero",
}
_TODOS_LOS_ALIAS = {a for alias in ALIAS_CULTIVO.values() for a in alias}


def extraer_cultivo_desconocido(texto):
    """Devuelve el nombre de un cultivo que la pregunta menciona y el sistema NO cubre, o
    None. Sólo tiene sentido llamarlo cuando `detectar_cultivos(texto)` ya dio vacío."""
    n = _norm(texto or "")
    for patron in _PATRONES_CULTIVO:
        for m in patron.finditer(n):
            palabra = m.group(1)
            if palabra in _NO_ES_CULTIVO or palabra in _TODOS_LOS_ALIAS:
                continue
            return palabra
    return None
