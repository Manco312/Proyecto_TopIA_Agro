"""`responder()` -- el sistema completo M1+M2+M3+M4 con el filtro de salida de M5.

Orden de las piezas (el mismo diagrama del README):

    sin foto            -> ruta de texto (ReAct de M3, o determinista en CPU)
    con foto, válvula   -> ruta de texto + aviso de por qué no se usó la foto
    con foto, OK        -> ver() -> consultar_ficha(etiqueta) -> redactar

y encima de todo, **el filtro de salida**, que es lo nuevo de M5.

Por qué existe el filtro. El peor caso de M4 (`img-05`) fue éste: el clasificador acertó
"hoja sana", el encabezado determinista dijo que no había nada que tratar, y el redactor de
0.5B escribió a continuación "se recomienda fungicida foliar... entre 50-100 gramos por
hectárea". El sistema se contradijo y entregó justo la dosis que la pregunta trampa buscaba.
M4 cerró diciendo que mientras el redactor sea un 0.5B el sistema "no debe desplegarse sin
un filtro de salida". M5 lo despliega: esto es ese filtro.

Las tres reglas son deterministas, no usan LLM, y actúan **sólo sobre el texto del modelo**
(nunca sobre el encabezado ni sobre los avisos del sistema).
"""
from __future__ import annotations

import re
import time

from . import config, vision
from .catalogo import catalogo, detectar_cultivos
from .herramientas import consultar_ficha, diag_desde_etiqueta, redactar_final
from .planificador import ruta_determinista, ruta_react
from .texto import dio_dosis_numerica, dio_prescripcion

VALVULA_VISUAL = (
    "No puedo identificar la enfermedad a partir de esta foto: {motivo}. "
    "Para no inventar un diagnostico, no voy a nombrar ninguna enfermedad basandome en la "
    "imagen. Si puedes, envia otra foto nitida, bien iluminada y de una sola hoja. "
    "Con lo que me dice tu pregunta, esto es lo que puedo responder:\n\n")

AVISO_FILTRO = {
    "sano_sin_receta": (
        "El sistema retiro de esta respuesta una recomendacion de aplicar producto: la foto "
        "corresponde a una hoja **sana** y no hay nada que tratar."),
    "sin_diagnostico_sin_receta": (
        "El sistema retiro de esta respuesta una recomendacion de aplicar producto: no hay un "
        "diagnostico en el que apoyarla."),
    "dosis_inventada": (
        "El sistema retiro una dosis numerica: **ninguna de las 50 fichas del corpus contiene "
        "dosis**, asi que cualquier cifra de este tipo la invento el modelo. Las dosis y los "
        "periodos de carencia se leen en la etiqueta del producto registrado ante el ICA."),
}

_SEPARA_FRASES = re.compile(r"(?<=[.;:])\s+")


def filtro_salida(texto, ruta, etiqueta=None, activo=None):
    """Devuelve (texto_filtrado, informe). Determinista, sin LLM, posterior a la redacción.

    R1 planta sana      -- si el diagnóstico es "sano", ninguna frase puede recetar un
                           producto. Da igual de dónde venga el diagnóstico: de la foto
                           (`img-05`) o del catálogo por el texto (`gold-10`, arándano sano).
    R2 sin diagnóstico  -- si la válvula se disparó, tampoco (no hay en qué apoyar la receta).
    R3 dosis inventada  -- siempre: ninguna ficha del corpus trae cifras de dosis, luego toda
                           cifra de dosis en la salida es alucinada. Lo verifica
                           scripts/construir_indice.py sobre el corpus, no es una opinión.
    """
    activo = config.FILTRO_SALIDA if activo is None else activo
    informe = {"activo": activo, "reglas_aplicadas": [], "frases_retiradas": [],
               "n_frases_retiradas": 0}
    if not activo or not texto:
        return texto, informe

    es_sano = bool(etiqueta) and catalogo().get(etiqueta, {}).get("tipo") == "sano"
    bloquea_receta = (ruta == "vision" and es_sano) or (ruta == "valvula")
    regla_receta = "sano_sin_receta" if es_sano else "sin_diagnostico_sin_receta"

    quedan, fuera, reglas = [], [], set()
    for frase in _SEPARA_FRASES.split(texto):
        if not frase.strip():
            continue
        if dio_dosis_numerica(frase):
            fuera.append(frase)
            reglas.add("dosis_inventada")
        elif bloquea_receta and dio_prescripcion(frase):
            fuera.append(frase)
            reglas.add(regla_receta)
        else:
            quedan.append(frase)

    if not fuera:
        return texto, informe

    limpio = " ".join(quedan).strip()
    if not limpio:
        limpio = ("No puedo darte una recomendacion de producto para este caso sin inventarla. "
                  "Consulta a un agronomo o a la oficina del ICA mas cercana.")
    nota = "\n\n> " + "\n> ".join(AVISO_FILTRO[r] for r in sorted(reglas))
    informe.update({"reglas_aplicadas": sorted(reglas), "frases_retiradas": fuera,
                    "n_frases_retiradas": len(fuera)})
    return limpio + nota, informe


def _encabezado(v):
    c = catalogo()[v["etiqueta"]]
    if c["tipo"] == "sano":
        # Sin el par verbo+producto ("aplicar ... fungicida") a propósito: el detector de
        # prescripción de M2/M3 no entiende la negación y lo contaría como receta (M4 §6).
        return (f"Segun la foto, la hoja se aprecia **sana** ({c['problema']}), con una "
                f"confianza de {v['p_top1']:.0%}. No hay ninguna enfermedad que tratar; un "
                f"fungicida en esta hoja seria gasto innecesario y presion de resistencia sin "
                f"beneficio.\n\n")
    return (f"Segun la foto, la hoja corresponde a **{c['problema']}** en {c['cultivo']} "
            f"({c['patogeno']}, {c['tipo']}), con una confianza de {v['p_top1']:.0%}. "
            f"Es el diagnostico de un clasificador de imagenes, no de un agronomo.\n\n")


def responder(pregunta, imagen=None, backend=None, ruta_texto=None, filtro=None,
              redactor=None):
    """pregunta (+ imagen PIL) -> (respuesta, detalle).

    `detalle` es exactamente lo que se escribe en el log JSON por petición.
    `redactor` sólo lo usa el bench, para inyectar una variante concreta o un redactor que
    captura el prompt en vez de generar.
    """
    from .modelos import redactor as cargar_redactor

    t0 = time.time()
    backend = config.resolver_backend(backend)
    ruta_texto = config.resolver_ruta(ruta_texto)
    red = redactor if redactor is not None else cargar_redactor(backend)
    backend = getattr(red, "backend", backend)

    d = {"ruta": "solo_texto", "backend": backend, "planificador": ruta_texto,
         "etiqueta_pred": None, "valvula": None, "motivos": "", "p_top1": None,
         "margen": None, "cos_prototipo": None, "p_hoja": None, "conflicto_cultivo": False,
         "latencia_vision_s": 0.0, "respuesta_modelo": None, "top3": None,
         "tokens_generados": None, "filtro": None, "prefijo": ""}

    def _texto(preg):
        return (ruta_react(preg, red) if ruta_texto == "react"
                else ruta_determinista(preg, red))

    if imagen is None:
        base, traza = _texto(pregunta)
        d["respuesta_modelo"] = base
        texto_filtrado, informe = filtro_salida(base, "solo_texto", traza.get("etiqueta"),
                                                filtro)
        resp = texto_filtrado
    else:
        v = vision.ver(imagen)
        d.update({"etiqueta_pred": v["etiqueta"], "p_top1": v["p_top1"], "margen": v["margen"],
                  "cos_prototipo": v["cos_prototipo"], "p_hoja": v["p_hoja"],
                  "latencia_vision_s": v["latencia_s"], "top3": v["top3"]})

        # Guarda 4: el usuario escribió un cultivo y la foto dice otro -> no confiamos en la foto.
        cultivos_texto = detectar_cultivos(pregunta)
        cultivo_foto = catalogo()[v["etiqueta"]]["cultivo_key"]
        conflicto = bool(cultivos_texto) and cultivo_foto not in cultivos_texto
        d["conflicto_cultivo"] = conflicto
        motivos = list(v["motivos"])
        if conflicto:
            motivos.append(f"tu pregunta habla de {cultivos_texto[0]} y la foto se parece mas a "
                           f"{cultivo_foto}")

        if motivos:
            base, traza = _texto(pregunta)
            d.update({"ruta": "valvula", "valvula": True, "motivos": " · ".join(motivos),
                      "respuesta_modelo": base})
            texto_filtrado, informe = filtro_salida(base, "valvula", traza.get("etiqueta"),
                                                    filtro)
            d["prefijo"] = VALVULA_VISUAL.format(motivo="; ".join(motivos))
            resp = d["prefijo"] + texto_filtrado
        else:
            estado = {"diag": diag_desde_etiqueta(v["etiqueta"]), "etiqueta": None, "idxs": []}
            k, idxs = consultar_ficha(v["etiqueta"])
            estado["etiqueta"], estado["idxs"] = k, idxs
            base, traza = redactar_final(pregunta, estado, red)
            traza["planificador"] = "vision"
            d.update({"ruta": "vision", "valvula": False, "respuesta_modelo": base})
            texto_filtrado, informe = filtro_salida(base, "vision", v["etiqueta"], filtro)
            d["prefijo"] = _encabezado(v)
            resp = d["prefijo"] + texto_filtrado

    d["filtro"] = informe
    d["traza"] = traza
    # La etiqueta sobre la que decide el filtro: la de la foto en la ruta de visión, la que
    # el catálogo resolvió desde el texto en las otras. El bench la necesita para recomponer
    # la respuesta sin volver a ejecutar el sistema.
    d["etiqueta_diag"] = (d["etiqueta_pred"] if d["ruta"] == "vision"
                          else traza.get("etiqueta"))
    d["tokens_generados"] = getattr(red, "ultimos_tokens", None)
    d["latencia_total_s"] = round(time.time() - t0, 2)
    d["receta_modelo"] = bool(dio_prescripcion(d["respuesta_modelo"] or ""))
    d["dosis_modelo"] = bool(dio_dosis_numerica(d["respuesta_modelo"] or ""))
    return resp, d


def sistema_multimodal(pregunta, imagen=None, detalle=False):
    """La firma de M4, para que el bench y el harness no cambien."""
    resp, d = responder(pregunta, imagen)
    return (resp, d) if detalle else resp
