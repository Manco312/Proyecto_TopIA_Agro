"""La técnica de eficiencia de M5, medida: cuantización del redactor.

El contrato vive en `bench_contrato.py` y **se declara antes de medir**. Este script sólo
genera; `puntuar.py` puntúa y emite el veredicto (van en procesos distintos porque el juez
de 1.5B no cabe en memoria junto a CLIP y el cross-encoder).

Cómo se aísla el efecto de la cuantización: los prompts se construyen UNA sola vez (pasada 1,
con un redactor que captura en vez de generar) y después **cada variante genera a partir de
exactamente los mismos prompts**. Así lo que se compara es el generador, no el retrieval ni
el clasificador de imágenes, que son idénticos en todas las variantes por construcción.

Uso:
    python scripts/bench_eficiencia.py --variantes fp32_cpu,int8_cpu --etiqueta cpu
    python scripts/bench_eficiencia.py --variantes fp16_gpu,int4_gpu --etiqueta gpu
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agro import config                                                   # noqa: E402


def cargar_casos(origen: Path, con_imagenes=True, eval_set="eval_set.json"):
    """Casos de texto + los 7 casos con imagen de M4. Los mismos de siempre: si el conjunto
    cambiara, la comparación con los scorecards anteriores no diría nada.

    `eval_set.json`     13 casos (10 gold + 3 adversariales), el conjunto de M2.
    `eval_set_m3.json`  27 casos (22 gold + 5 adversariales), el de M3 y M4 -- es el que hay
                        que usar para que el scorecard sea comparable con los suyos.
    """
    casos = []
    with open(origen / eval_set, encoding="utf-8") as f:
        for c in json.load(f):
            casos.append({**c, "fuente": Path(eval_set).stem, "imagen": None})

    ruta_csv = origen / "eval_set" / "casos_m4.csv"
    if con_imagenes and ruta_csv.exists():
        with open(ruta_csv, encoding="utf-8", newline="") as f:
            for fila in csv.DictReader(f):
                p = origen / "eval_set" / "imagenes" / fila["imagen"]
                if not p.exists():
                    continue
                casos.append({
                    "id": fila["id"], "tipo": fila["tipo"], "input": fila["pregunta"],
                    "esperado": fila["esperado"],
                    "palabras_clave": [k for k in fila["palabras_clave"].split("|") if k],
                    "patogeno_esperado": fila["patogeno_esperado"] or None,
                    "label_plantvillage": fila["label_plantvillage"] or None,
                    "cultivo_en": fila["cultivo_en"] or None,
                    "categoria_adversarial": fila["categoria_adversarial"] or None,
                    "criterio": fila["criterio_texto"], "fuente": "imagen_m4",
                    "imagen": str(p)})
    return casos


class Captura:
    """Redactor que no genera: guarda el prompt que le habrían pasado."""
    backend, ultimos_tokens = "captura", 0

    def __init__(self):
        self.prompts = []

    def generar(self, prompt, max_new=config.MAX_NEW_SISTEMA):
        self.prompts.append(prompt)
        return "«GENERACION»"


class Reproduce:
    """Redactor que devuelve un texto ya generado, para recomponer la respuesta final con
    su encabezado, su aviso de válvula y su filtro de salida."""
    ultimos_tokens = 0

    def __init__(self, texto, backend):
        self.texto, self.backend = texto, backend

    def generar(self, prompt, max_new=config.MAX_NEW_SISTEMA):
        return self.texto


def preparar(casos, ruta_texto):
    """Pasada 1. Deja grabado todo lo que hace falta para recomponer la respuesta final sin
    volver a tocar un modelo: el prompt, el prefijo determinista (encabezado de la foto o
    aviso de válvula), la ruta y la etiqueta que vio el clasificador."""
    from PIL import Image

    from agro.sistema import responder
    preparados = []
    for i, c in enumerate(casos, 1):
        img = Image.open(c["imagen"]).convert("RGB") if c["imagen"] else None
        cap = Captura()
        _, d = responder(c["input"], img, ruta_texto=ruta_texto, redactor=cap)
        preparados.append({
            "caso_id": c["id"], "prompt": cap.prompts[0] if cap.prompts else None,
            "ruta": d["ruta"], "etiqueta_pred": d["etiqueta_pred"], "p_top1": d["p_top1"],
            "valvula": d["valvula"], "motivos": d["motivos"], "prefijo": d["prefijo"],
            "etiqueta_diag": d["etiqueta_diag"],
            "texto_fijo": None if cap.prompts else d["respuesta_modelo"],
            "sin_llm": not cap.prompts, "imagen": c["imagen"], "input": c["input"],
            "tipo": c["tipo"], "fuente": c["fuente"]})
        marca = "sin LLM" if not cap.prompts else f"prompt de {len(cap.prompts[0])} chars"
        print(f"   [{i:2d}/{len(casos)}] {c['id'][:44]:44s} ruta={d['ruta']:10s} {marca}")
    return preparados


def medir_variante(backend, preparados):
    """Pasada 2. Sólo el redactor está vivo: la recomposición es string + filtro, sin CLIP,
    sin reranker y sin embebedor. Antes se volvía a llamar a `responder()` por caso y eso
    mantenía ~1.5 GB de modelos cargados que no se usaban, con la máquina de medición en
    swap -- y una latencia medida bajo swap no mide la cuantización, mide el disco."""
    from agro.modelos import redactor, soltar_redactor
    from agro.sistema import filtro_salida

    red = redactor(backend)
    info = red.info()
    print(f"\n=== {backend} === {json.dumps(info, ensure_ascii=False)}")

    filas = []
    for i, p in enumerate(preparados, 1):
        if p["sin_llm"]:
            texto_modelo, seg, toks = p["texto_fijo"] or "", 0.0, 0
        else:
            t0 = time.time()
            texto_modelo = red.generar(p["prompt"])
            seg = time.time() - t0
            toks = getattr(red, "ultimos_tokens", 0) or 0

        filtrado, informe = filtro_salida(texto_modelo, p["ruta"], p["etiqueta_diag"])
        filas.append({
            "backend": backend, "id": p["caso_id"], "tipo": p["tipo"], "fuente": p["fuente"],
            "ruta": p["ruta"], "sin_llm": p["sin_llm"],
            "segundos_generacion": round(seg, 3), "tokens": toks,
            "tokens_por_s": round(toks / seg, 2) if seg > 0.01 else None,
            "etiqueta_pred": p["etiqueta_pred"], "etiqueta_diag": p["etiqueta_diag"],
            "filtro_n": informe["n_frases_retiradas"],
            "filtro_reglas": "|".join(informe["reglas_aplicadas"]),
            "respuesta_modelo": texto_modelo,
            "respuesta_final": p["prefijo"] + filtrado,
        })
        print(f"   [{i:2d}/{len(preparados)}] {p['caso_id'][:40]:40s} {seg:6.2f} s  "
              f"{toks:3d} tok  filtro={informe['n_frases_retiradas']}")
    del red                       # sin esto el modelo sigue vivo en este marco de pila y
    soltar_redactor(backend)      # la siguiente variante se queda sin memoria
    return filas, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variantes", default="fp32_cpu,int8_cpu")
    ap.add_argument("--ruta", default="determinista")
    ap.add_argument("--origen", default=str(config.ORIGEN_DATOS))
    ap.add_argument("--eval-set", default="eval_set.json",
                    help="eval_set.json (13 casos de M2) o eval_set_m3.json (27, el de M3/M4)")
    ap.add_argument("--etiqueta", default="cpu")
    ap.add_argument("--sin-imagenes", action="store_true")
    ap.add_argument("--sin-juez", action="store_true")
    ap.add_argument("--base", default="", help="variante que hace de 'antes' (por defecto "
                                               "fp32_cpu); se pasa a puntuar.py")
    ap.add_argument("--merge", default="",
                    help="crudo anterior al que añadir estas variantes. La pasada 1 es "
                         "determinista, así que los prompts son los mismos y las filas son "
                         "comparables sin volver a generar las variantes ya medidas.")
    ap.add_argument("--no-puntuar", action="store_true")
    ap.add_argument("--hilos", type=int, default=0,
                    help="limitar los hilos de torch. Con --hilos 2 la latencia medida es "
                         "la que cabe esperar en un Space CPU basic (2 vCPU).")
    ap.add_argument("--limite", type=int, default=0, help="sólo para depurar el script")
    args = ap.parse_args()

    if args.hilos:
        import torch
        torch.set_num_threads(args.hilos)

    variantes = [v.strip() for v in args.variantes.split(",") if v.strip()]
    origen = Path(args.origen)
    config.RUTA_TEXTO = args.ruta
    config.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

    from agro.modelos import fijar_semillas, revision
    fijar_semillas()

    casos = cargar_casos(origen, not args.sin_imagenes, args.eval_set)
    if args.limite:
        casos = casos[:args.limite]
    print(f"{len(casos)} casos ({sum(c['tipo'] == 'gold' for c in casos)} gold + "
          f"{sum(c['tipo'] == 'adversarial' for c in casos)} adversariales; "
          f"{sum(bool(c['imagen']) for c in casos)} con foto)\n")

    print("Pasada 1: construyendo los prompts una sola vez ...")
    preparados = preparar(casos, args.ruta)
    con_gen = sum(not p["sin_llm"] for p in preparados)
    print(f"   {con_gen} casos llegan al generador; {len(casos) - con_gen} se resuelven sin "
          f"LLM (abstención o aclaración deterministas).\n")

    # A partir de aquí sólo hace falta el redactor: soltamos CLIP, el embebedor y el
    # cross-encoder para que la medición de latencia no la haga una máquina en swap.
    import gc

    from agro import modelos
    modelos._cache.clear()
    gc.collect()

    todas, infos = [], {}
    for backend in variantes:
        try:
            filas, info = medir_variante(backend, preparados)
        except Exception as e:                                            # noqa: BLE001
            print(f"   {backend}: NO SE PUDO MEDIR -> {type(e).__name__}: {e}")
            continue
        infos[backend] = info
        todas.extend(filas)

    entorno = {"python": platform.python_version(), "sistema": platform.system(),
               "procesador": platform.processor(), "cpus": __import__("os").cpu_count(),
               "hilos_torch": __import__("torch").get_num_threads()}
    for mod in ("torch", "transformers", "peft", "sklearn", "numpy"):
        try:
            entorno[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            entorno["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    if args.merge:
        with open(args.merge, encoding="utf-8") as f:
            previo = json.load(f)
        nuevas = set(infos)
        todas = [x for x in previo["filas"] if x["backend"] not in nuevas] + todas
        infos = {**previo["infos"], **infos}
        print(f"   fusionado con {args.merge}: variantes {list(infos)}")

    crudo = config.DIR_RESULTADOS / f"eficiencia_{args.etiqueta}_crudo.json"
    with open(crudo, "w", encoding="utf-8") as f:
        json.dump({"etiqueta": args.etiqueta,
                   "fecha": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "ruta_texto": args.ruta, "entorno": entorno,
                   "revisiones": {config.MODEL_BASE_ID: revision(config.MODEL_BASE_ID),
                                  config.PLANIFICADOR_ID: revision(config.PLANIFICADOR_ID)},
                   "preparacion": [{k: v for k, v in p.items() if k != "prompt"}
                                   for p in preparados],
                   "infos": infos, "casos": casos, "filas": todas},
                  f, ensure_ascii=False, indent=1)
    print(f"\nGeneración terminada -> {crudo}")

    if args.no_puntuar:
        print("(--no-puntuar: corre `python scripts/puntuar.py "
              f"{crudo.relative_to(config.RAIZ)}` cuando quieras)")
        return 0

    # El hijo carga un 1.5B: si el padre sigue ocupando memoria, Windows mata la carga con
    # "el archivo de paginación es demasiado pequeño" (os error 1455).
    modelos._cache.clear()
    gc.collect()

    cmd = [sys.executable, "-X", "utf8", str(Path(__file__).parent / "puntuar.py"), str(crudo)]
    if args.sin_juez:
        cmd.append("--sin-juez")
    if args.base:
        cmd += ["--base", args.base]
    print(f"\nPuntuando en un proceso aparte: {' '.join(cmd[2:])}\n")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
