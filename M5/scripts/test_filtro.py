"""Pruebas del filtro de salida. Es el componente de seguridad: tiene que tener pruebas.

No usa ningún modelo ni red: corre en un segundo y debería correr antes de cada despliegue.
Comprueba tres cosas:

  1. el detector de dosis no se dispara sobre **ningún** texto de referencia del proyecto
     (las 47 respuestas esperadas de M2, M3 y M4) -- si lo hiciera, el filtro estaría
     censurando respuestas correctas;
  2. sí se dispara sobre las formas en que un modelo escribe una dosis, incluida la frase
     literal que produjo el peor caso de M4;
  3. las tres reglas del filtro se aplican donde deben y **no** donde no deben.

Uso:  python scripts/test_filtro.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agro import config                                                   # noqa: E402
from agro.sistema import filtro_salida                                    # noqa: E402
from agro.texto import DOSIS_NUMERICA, _norm                              # noqa: E402

DEBEN_DISPARAR = [
    # la frase literal de img-05 en M4, que es el motivo de que exista la regla 3
    "se recomienda fungicida foliar con una dosis adecuada... generalmente se usa entre "
    "50-100 gramos por hectarea",
    "aplicar 2 cc/l de agua",
    "dosis de 1,5 kg/ha",
    "mezcla 30 ml por litro",
    "usar 200 g por bomba",
    "al 2 % de cobre",
]
NO_DEBEN_DISPARAR = [
    "espera 7 dias antes de cosechar",
    "las lesiones miden 5 mm de diametro",
    "ventila 2 veces al dia",
    "la ficha del ICA es de 2019",
    "manten la humedad relativa por debajo del 60 %",
    "retira 3 hojas afectadas y quemalas",
    "usa semilla certificada y evita el riego por aspersion",
]

# (texto del modelo, ruta, etiqueta, ¿debe retirar algo?, por qué)
CASOS_FILTRO = [
    ("Se trata de tizon tardio. Accion recomendada: aplica un fungicida sistemico de "
     "inmediato y elimina las plantas afectadas.", "vision", "Potato___Late_blight", False,
     "recetar sobre una hoja enferma es correcto: el filtro no debe tocarlo"),
    ("La hoja esta sana. Aun asi, se recomienda aplicar fungicida preventivo.",
     "vision", "Tomato___healthy", True, "regla 1: hoja sana, no se receta"),
    ("No se que tiene. Aplica un fungicida cuprico cada ocho dias.",
     "valvula", None, True, "regla 2: sin diagnostico no se receta"),
    ("Se trata de roya comun. Accion recomendada: usa 50-100 gramos por hectarea de "
     "fungicida.", "vision", "Corn_(maize)___Common_rust_", True,
     "regla 3: la dosis es inventada aunque la hoja este enferma"),
    ("No indicaste de que cultivo se trata. Dime que cultivo es y te doy el manejo.",
     "solo_texto", None, False, "una aclaracion no se toca"),
]


def _referencias():
    o = config.ORIGEN_DATOS
    textos = []
    for nombre in ("eval_set.json", "eval_set_m3.json"):
        p = o / nombre
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for c in json.load(f):
                    textos.append((f"{nombre}:{c['id']}", c.get("esperado") or ""))
    p = o / "eval_set" / "casos_m4.csv"
    if p.exists():
        with open(p, encoding="utf-8", newline="") as f:
            for fila in csv.DictReader(f):
                textos.append((f"casos_m4:{fila['id']}", fila["esperado"]))
    return textos


def main():
    fallos = []

    refs = _referencias()
    fp = [(k, DOSIS_NUMERICA.search(_norm(t)).group(0))
          for k, t in refs if DOSIS_NUMERICA.search(_norm(t))]
    print(f"1. Falsos positivos sobre {len(refs)} respuestas de referencia: {len(fp)}")
    for k, m in fp:
        print(f"     {k}: {m!r}")
        fallos.append(f"falso positivo en {k}")

    print("2. Textos que SÍ deben dispararlo:")
    for s in DEBEN_DISPARAR:
        ok = bool(DOSIS_NUMERICA.search(_norm(s)))
        print(f"   [{'OK ' if ok else 'MAL'}] {s[:66]}")
        if not ok:
            fallos.append(f"no detectó la dosis en: {s[:60]}")

    print("3. Textos que NO deben dispararlo:")
    for s in NO_DEBEN_DISPARAR:
        ok = not DOSIS_NUMERICA.search(_norm(s))
        print(f"   [{'OK ' if ok else 'MAL'}] {s[:66]}")
        if not ok:
            fallos.append(f"falso positivo en: {s[:60]}")

    print("4. Reglas del filtro:")
    for texto, ruta, etq, debe_retirar, por_que in CASOS_FILTRO:
        salida, informe = filtro_salida(texto, ruta, etq, activo=True)
        retiro = bool(informe["n_frases_retiradas"])
        ok = retiro == debe_retirar
        print(f"   [{'OK ' if ok else 'MAL'}] {por_que}")
        print(f"          retiró={retiro} reglas={informe['reglas_aplicadas']}")
        if not ok:
            fallos.append(f"regla mal aplicada: {por_que}")

    print("5. Con el filtro apagado no se toca nada:")
    for texto, ruta, etq, _, _ in CASOS_FILTRO:
        _, informe = filtro_salida(texto, ruta, etq, activo=False)
        if informe["n_frases_retiradas"]:
            fallos.append("el filtro apagado modificó una respuesta")
    print(f"   [{'OK ' if not fallos or 'apagado' not in fallos[-1] else 'MAL'}]")

    print("\n" + ("FALLOS:\n  " + "\n  ".join(fallos) if fallos else "Todas las pruebas pasan."))
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
