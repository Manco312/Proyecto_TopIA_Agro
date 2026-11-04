"""Ablación 2 — ¿qué se pierde al quitar el planificador ReAct de M3?

Es la decisión de arquitectura que hace posible el despliegue en CPU: el bucle ReAct usa un
Qwen2.5-1.5B para elegir herramientas, y en 2 vCPU eso cuesta minutos por consulta. La ruta
determinista fija en código la secuencia que el ReAct recorre casi siempre (diagnóstico
diferencial -> ficha de la etiqueta más probable -> redactar).

Las dos rutas se corren con **el mismo redactor** y sobre los mismos casos de texto, así
que lo que se compara es el planificador. Se mide latencia de punta a punta (que es donde
está el efecto), tipo de salida y las métricas de M2/M3.

Uso:
    python scripts/bench_ablacion.py --backend fp16_gpu
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench_eficiencia import cargar_casos                                 # noqa: E402

from agro import config                                                   # noqa: E402

RUTAS = ["react", "determinista"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None)
    ap.add_argument("--origen", default=str(config.ORIGEN_DATOS))
    ap.add_argument("--eval-set", default="eval_set_m3.json")
    ap.add_argument("--etiqueta", default="planificador")
    ap.add_argument("--sin-juez", action="store_true")
    ap.add_argument("--limite", type=int, default=0)
    args = ap.parse_args()

    backend = config.resolver_backend(args.backend)
    config.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

    from agro.modelos import fijar_semillas, redactor
    from agro.sistema import responder
    fijar_semillas()

    # Sólo casos de texto: con foto y clasificador seguro el sistema NO pasa por el
    # planificador (M4 §5), así que incluirlos diluiría el efecto que se quiere medir.
    casos = cargar_casos(Path(args.origen), con_imagenes=False, eval_set=args.eval_set)
    if args.limite:
        casos = casos[:args.limite]
    print(f"{len(casos)} casos de texto · redactor={backend}\n")

    red = redactor(backend)
    filas, infos = [], {}
    for ruta in RUTAS:
        print(f"=== ruta = {ruta} ===")
        for i, c in enumerate(casos, 1):
            t0 = time.time()
            resp, d = responder(c["input"], None, ruta_texto=ruta, redactor=red)
            seg = time.time() - t0
            traza = d.get("traza") or {}
            pasos = traza.get("pasos") or []
            filas.append({
                "backend": ruta, "id": c["id"], "tipo": c["tipo"], "fuente": c["fuente"],
                "ruta": d["ruta"], "sin_llm": False,
                "segundos_generacion": round(seg, 3),
                "tokens": d.get("tokens_generados") or 0,
                "tokens_por_s": None, "etiqueta_pred": None,
                "etiqueta_diag": d.get("etiqueta_diag"),
                "salida": traza.get("salida"), "n_pasos": len(pasos),
                "herramientas": "|".join(p["herramienta"] for p in pasos),
                "filtro_n": d["filtro"]["n_frases_retiradas"],
                "filtro_reglas": "|".join(d["filtro"]["reglas_aplicadas"]),
                "respuesta_modelo": d["respuesta_modelo"] or "",
                "respuesta_final": resp,
            })
            print(f"   [{i:2d}/{len(casos)}] {c['id'][:40]:40s} {seg:7.2f} s  "
                  f"salida={traza.get('salida')}  pasos={len(pasos)}")
        infos[ruta] = {"backend": ruta, "dtype": f"planificador={ruta}",
                       "dispositivo": red.info()["dispositivo"],
                       "mib_pesos": red.info()["mib_pesos"], "capas_cuantizadas": 0,
                       "segundos_carga": red.info()["segundos_carga"]}

    entorno = {"python": platform.python_version(), "sistema": platform.system(),
               "redactor": backend, "planificador_id": config.PLANIFICADOR_ID}
    try:
        import torch
        entorno["hilos_torch"] = torch.get_num_threads()
        if torch.cuda.is_available():
            entorno["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    crudo = config.DIR_RESULTADOS / f"ablacion_{args.etiqueta}_crudo.json"
    with open(crudo, "w", encoding="utf-8") as f:
        json.dump({"etiqueta": args.etiqueta, "fecha": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "ruta_texto": "ambas", "entorno": entorno, "infos": infos,
                   "preparacion": [], "casos": casos, "filas": filas},
                  f, ensure_ascii=False, indent=1)
    print(f"\nGeneración terminada -> {crudo}")

    import gc

    from agro import modelos
    del red
    modelos._cache.clear()
    gc.collect()

    cmd = [sys.executable, "-X", "utf8", str(Path(__file__).parent / "puntuar.py"), str(crudo)]
    if args.sin_juez:
        cmd.append("--sin-juez")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
