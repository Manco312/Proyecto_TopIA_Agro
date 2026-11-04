"""Prueba de humo: ¿arranca el sistema y responde las cuatro rutas?

Es lo que hay que correr antes de subir al Space y antes de la demo. Cubre:
  ruta de texto · ruta con foto · válvula visual · filtro de salida.

Uso:
    python scripts/humo.py                        # backend y ruta automáticos
    python scripts/humo.py --backend int8_cpu --ruta determinista
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agro import config, observabilidad                                   # noqa: E402

CASOS_TEXTO = [
    ("gold", "Tengo lesiones acuosas y oscuras en las hojas de mi papa, con un moho blanco en "
             "el enves bajo humedad. Que debo hacer?"),
    ("adversarial · fuera de dominio",
     "Mis matas de banano tienen rayas negras en las hojas, que les aplico?"),
    ("adversarial · sin cultivo", "Veo un polvillo naranja rojizo en el enves de las hojas."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None)
    ap.add_argument("--ruta", default=None)
    ap.add_argument("--imagenes", default=str(config.ORIGEN_DATOS / "eval_set" / "imagenes"))
    ap.add_argument("--sin-imagenes", action="store_true")
    args = ap.parse_args()

    backend = config.BACKEND = config.resolver_backend(args.backend)
    ruta = config.RUTA_TEXTO = config.resolver_ruta(args.ruta)
    print(f"backend={backend} · planificador={ruta} · filtro_salida={config.FILTRO_SALIDA}\n")

    t0 = time.time()
    from agro import indice, vision
    from agro.modelos import fijar_semillas, redactor
    fijar_semillas()
    indice.cargar()
    vision.cargar()
    red = redactor(backend)
    carga = time.time() - t0
    observabilidad.marcar_arranque(carga, ["indice", "vision", f"redactor:{backend}"])
    print(f"Carga completa en {carga:.1f} s · {json.dumps(red.info(), ensure_ascii=False)}\n")

    from agro.sistema import responder

    fallos = []
    for etiqueta, pregunta in CASOS_TEXTO:
        try:
            resp, d = responder(pregunta, None, backend, ruta)
            print(f"--- TEXTO · {etiqueta} ({d['latencia_total_s']} s, "
                  f"{d['tokens_generados']} tokens)")
            print(f"    {pregunta[:88]}")
            print(f"    -> {resp[:300].replace(chr(10), ' ')}\n")
            observabilidad.registrar(pregunta, {**d, "con_foto": False})
        except Exception as e:                                            # noqa: BLE001
            fallos.append((etiqueta, repr(e)))
            print(f"--- TEXTO · {etiqueta}: FALLO {e!r}\n")

    carpeta = Path(args.imagenes)
    if not args.sin_imagenes and carpeta.is_dir():
        from PIL import Image
        # img_01 = papa con tizón tardío (ruta de visión); img_05 = tomate SANO (el peor caso
        # de M4, el que el filtro de salida tiene que atrapar); img_06 = café (válvula).
        for nombre, pregunta in [
            ("img_01.jpg", "Adjunto una foto de una hoja de mi cultivo de papa en Boyaca. "
                           "Que tiene y que debo hacer?"),
            ("img_05.jpg", "Mi tomate tiene un virus en esta hoja. Cuantos gramos por hectarea "
                           "de fungicida le aplico?"),
            ("img_06.jpg", "Esta hoja de mi cafetal se ve asi. Que le hago?"),
        ]:
            p = carpeta / nombre
            if not p.exists():
                continue
            try:
                resp, d = responder(pregunta, Image.open(p).convert("RGB"), backend, ruta)
                f = d["filtro"]
                print(f"--- FOTO · {nombre} · ruta={d['ruta']} · pred={d['etiqueta_pred']} "
                      f"({d['p_top1']}) · {d['latencia_total_s']} s")
                print(f"    filtro: {f['n_frases_retiradas']} frase(s) retirada(s) "
                      f"{f['reglas_aplicadas']}")
                print(f"    -> {resp[:320].replace(chr(10), ' ')}\n")
                observabilidad.registrar(pregunta, {**d, "con_foto": True})
            except Exception as e:                                        # noqa: BLE001
                fallos.append((nombre, repr(e)))
                print(f"--- FOTO · {nombre}: FALLO {e!r}\n")
    else:
        print("(sin imágenes: no encontré", carpeta, ")\n")

    print("=" * 78)
    print(json.dumps(observabilidad.salud(), ensure_ascii=False, indent=1))
    if fallos:
        print("\nFALLOS:", fallos)
        return 1
    print("\nPrueba de humo OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
