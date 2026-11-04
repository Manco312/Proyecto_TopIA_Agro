"""Las dos rutas del camino sin foto.

`ruta_react` es el mini-agente de M3 §14 sin cambios: un Qwen2.5-1.5B escribe
`Pensamiento:` / `Acción:` y el bucle ejecuta la herramienta que pida, hasta 4 pasos.

`ruta_determinista` hace la misma secuencia **sin modelo planificador**: diagnóstico
diferencial -> ficha de la etiqueta más probable -> redactar. Existe porque en 2 vCPU el
1.5B cuesta minutos por consulta; la ablación del reporte mide qué se pierde al quitarlo.
"""
from __future__ import annotations

import re

from .catalogo import detectar_cultivos, extraer_cultivo_desconocido
from .config import MAX_PASOS_REACT
from .herramientas import (consultar_ficha, diagnostico_diferencial, ejecutar_herramienta,
                           redactar_final)

SYSTEM_REACT = (
    "Eres el planificador de un recomendador agronomico. Eliges herramientas; NO redactas la respuesta final.\n"
    "Herramientas:\n"
    "- diagnostico_diferencial[cultivo | sintomas]: dice si el cultivo esta cubierto y lista enfermedades "
    "candidatas con su etiqueta. Si la pregunta no dice el cultivo, deja cultivo vacio.\n"
    "- consultar_ficha[etiqueta]: trae la ficha tecnica de una etiqueta (ej. Potato___Late_blight).\n"
    "- buscar_en_fichas[consulta]: busqueda libre en todas las fichas, si lo anterior no alcanza.\n"
    "- Responder[ok]: cuando ya tengas la ficha correcta, o si el diagnostico dice que el cultivo no esta "
    "cubierto o que falta el cultivo.\n"
    "En cada paso escribe exactamente dos lineas:\nPensamiento: <razonamiento breve>\n"
    "Accion: <herramienta>[<argumentos>]\n"
    "Orden recomendado: diagnostico_diferencial, luego consultar_ficha con la etiqueta mas probable, luego Responder[ok].\n"
    "Ejemplo:\nPregunta: Mi fresa tiene manchas purpuras en las hojas.\n"
    "Pensamiento: Primero verifico el cultivo y las enfermedades candidatas.\n"
    "Accion: diagnostico_diferencial[fresa | manchas purpuras en las hojas]")

_RE_ACCION = re.compile(
    r"(diagnostico_diferencial|consultar_ficha|buscar_en_fichas|Responder)\[(.*?)\]")


def _paso_react(historial):
    from .modelos import generar_agente
    salida = generar_agente(SYSTEM_REACT, historial, 120)
    linea = next((l for l in salida.splitlines() if "Acci" in l or "Responder[" in l), salida)
    m = _RE_ACCION.search(linea) or _RE_ACCION.search(salida)
    return (salida, (m.group(1), m.group(2).strip())) if m else (salida, None)


def _args_desde_texto(nombre, arg):
    if nombre == "diagnostico_diferencial":
        cult, sep, sint = arg.partition("|")
        return {"cultivo": cult.strip(), "sintomas": sint.strip()} if sep else {"cultivo": "",
                                                                                "sintomas": arg}
    if nombre == "consultar_ficha":
        return {"etiqueta": arg}
    return {"consulta": arg}


def ruta_react(pregunta, redactor, max_pasos=MAX_PASOS_REACT):
    estado = {"diag": None, "etiqueta": None, "idxs": []}
    hist = f"Pregunta: {pregunta}"
    pasos = []
    for _ in range(max_pasos):
        _, accion = _paso_react(hist)
        if accion is None or accion[0] == "Responder":
            pasos.append({"herramienta": "Responder", "arg": "ok"})
            break
        nombre, arg = accion
        obs = ejecutar_herramienta(nombre, _args_desde_texto(nombre, arg), pregunta, estado)
        pasos.append({"herramienta": nombre, "arg": arg[:80], "observacion": obs[:120]})
        hist += f"\nAccion: {nombre}[{arg}]\nObservacion: {obs}"
        if estado["diag"] and estado["diag"]["estado"] != "cubierto":
            break
    texto, traza = redactar_final(pregunta, estado, redactor)
    traza.update({"planificador": "react", "pasos": pasos})
    return texto, traza


def ruta_determinista(pregunta, redactor):
    """La misma secuencia que el ReAct recorre en el 90 % de los casos, pero fijada en
    código: no hay modelo que decida el orden, así que no hay 1.5B que cargar.

    El único papel del planificador que hay que reponer a mano es sacar de la pregunta un
    cultivo que el sistema NO cubre ("mis matas de banano"), para poder abstenerse nombrando
    el límite en vez de pedir aclaración. Ver `catalogo.extraer_cultivo_desconocido`.
    """
    arg_cultivo = ""
    if not detectar_cultivos(pregunta):
        arg_cultivo = extraer_cultivo_desconocido(pregunta) or ""
    estado = {"diag": diagnostico_diferencial(arg_cultivo, pregunta),
              "etiqueta": None, "idxs": []}
    pasos = [{"herramienta": "diagnostico_diferencial", "arg": arg_cultivo or "(automático)",
              "observacion": estado["diag"]["estado"]}]
    if estado["diag"]["estado"] == "cubierto" and estado["diag"]["candidatos"]:
        etq = estado["diag"]["candidatos"][0]["etiqueta"]
        k, idxs = consultar_ficha(etq)
        if k is not None:
            estado["etiqueta"], estado["idxs"] = k, list(idxs)
            pasos.append({"herramienta": "consultar_ficha", "arg": k,
                          "observacion": f"{len(idxs)} pasajes"})
    texto, traza = redactar_final(pregunta, estado, redactor)
    traza.update({"planificador": "determinista", "pasos": pasos})
    return texto, traza
