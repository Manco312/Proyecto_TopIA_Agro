"""Puntúa las respuestas crudas de un bench y decide si la variante cumple el contrato.

Va en un proceso aparte a propósito. El juez de M2 es un Qwen2.5-1.5B y en la máquina donde
se midió esto no cabe en memoria junto a CLIP, el embebedor y el cross-encoder: el proceso
moría sin excepción. Separarlo tiene además una ventaja: volver a puntuar (p. ej. tras
corregir una métrica) no obliga a volver a generar nada.

Uso:
    python scripts/puntuar.py resultados/eficiencia_cpu_crudo.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Antes de que nadie importe torch. El cargador de transformers reserva de golpe un bloque
# contiguo del tamaño del modelo ("caching_allocator_warmup") y en una GPU de 6 GB eso falla
# aunque haya sitio de sobra sumando huecos; con segmentos expandibles no hace falta que el
# bloque sea contiguo.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench_contrato import CONTRATO, LINEA_BASE, evaluar_contrato, resumir   # noqa: E402

from agro.catalogo import catalogo                                           # noqa: E402
from agro.texto import (acierto_dominio, dio_dosis_numerica,                 # noqa: E402
                        dio_prescripcion)


def puntuar(filas, casos_por_id, todos, con_juez=True):
    """Métricas de M2/M3 sobre cada respuesta.

    Dimensión 1 sobre `respuesta_modelo` y no sobre la respuesta compuesta: es la corrección
    que M4 §9 dejó pedida al comprobar que el encabezado determinista bajaba la similitud
    entre 0.018 y 0.242 y cruzaba el umbral en dos casos.
    """
    import gc

    from agro import modelos
    from agro.juez import sim_embeddings

    # Pasada A: las similitudes, que sólo necesitan el embebedor (470 MB).
    print("   dimensión 1 (similitud de embeddings) ...")
    for f in filas:
        c = casos_por_id[f["id"]]
        texto = f["respuesta_modelo"] or f["respuesta_final"]
        f["sim"] = (round(sim_embeddings(texto, c["esperado"]), 4)
                    if c.get("esperado") else None)

    # El juez es un 1.5B y en la máquina de medición no cabe junto al embebedor: se suelta
    # antes de cargarlo. Sin esto el proceso muere sin excepción al mapear los pesos.
    modelos._cache.clear()
    gc.collect()

    # Pasada B: el juez.
    if con_juez:
        print("   dimensión 2 (juez de M2, un forward por respuesta) ...")
        from agro.juez import juez_puntua
    for n, f in enumerate(filas, 1):
        c = casos_por_id[f["id"]]
        if con_juez:
            s, e, _ = juez_puntua(c, f["respuesta_final"])
            f["juez"], f["juez_entropia"] = round(s, 3), round(e, 3)
        else:
            f["juez"], f["juez_entropia"] = 0.0, None
        a = acierto_dominio(c, f["respuesta_final"], f["sim"] or 0.0, f["juez"], todos)
        f["acierto"] = bool(a["acierto"])
        f["formato_ok"] = bool(a["formato_ok"])
        f["abstuvo"] = bool(a.get("abstuvo"))
        f["receta_bruta"] = dio_prescripcion(f["respuesta_modelo"] or "")
        f["dosis_bruta"] = dio_dosis_numerica(f["respuesta_modelo"] or "")
        f["receta_final"] = dio_prescripcion(f["respuesta_final"])
        f["dosis_final"] = dio_dosis_numerica(f["respuesta_final"])
        # El diagnóstico sobre el que decide el filtro: el de la foto en la ruta de
        # visión, el que el catálogo resolvió desde el texto en las demás.
        et = f.get("etiqueta_diag") or f["etiqueta_pred"]
        f["sobre_hoja_sana"] = bool(et and catalogo().get(et, {}).get("tipo") == "sano")
        f["receta_sobre_sana"] = bool(f["sobre_hoja_sana"] and f["receta_final"])
        if n % 10 == 0:
            print(f"   {n}/{len(filas)}")
    return filas


def tabla_markdown(salida):
    base = salida["linea_base"]
    r0 = salida["variantes"][base]
    cab = ("| variante | dónde | precisión | pesos | p50 gen. | p95 gen. | tok/s | "
           "sim gold | aciertos gold | juez | dosis en salida | contrato |\n"
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    filas = []
    for b, r in salida["variantes"].items():
        lat = r["latencia_generacion_s"]
        frac = f" ({r['mib_pesos'] / r0['mib_pesos']:.2f}×)" if r0["mib_pesos"] else ""
        v = salida["veredictos"].get(b)
        marca = ("— (es el «antes»)" if b == base else
                 "—" if v is None else
                 "✅ cumple" if v["cumple_contrato"] else "❌ NO cumple")
        filas.append(
            f"| `{b}`{' *(referencia)*' if b == base else ''} | {r['dispositivo']} | "
            f"{r['dtype']} | {r['mib_pesos']:.0f} MiB{frac} | {lat['p50']} s | {lat['p95']} s | "
            f"{r['tokens_por_s_media']} | {r['calidad']['sim_media_gold']} | "
            f"{r['calidad']['aciertos_gold']} | {r['calidad']['juez_medio']} | "
            f"{r['seguridad']['dosis_en_la_salida']} | {marca} |")
    return cab + "\n".join(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("crudo")
    ap.add_argument("--sin-juez", action="store_true")
    ap.add_argument("--base", default=LINEA_BASE,
                    help="variante que hace de 'antes'. Por defecto fp32_cpu, que es el "
                         "mismo modelo sin cuantizar en el mismo sitio donde se despliega.")
    args = ap.parse_args()

    ruta = Path(args.crudo)
    with open(ruta, encoding="utf-8") as f:
        bruto = json.load(f)

    filas = bruto["filas"]
    casos = bruto["casos"]
    casos_por_id = {c["id"]: c for c in casos}
    print(f"Puntuando {len(filas)} respuestas de {len(bruto['infos'])} variantes"
          f"{'' if args.sin_juez else ' (con el juez de M2)'} ...")
    puntuar(filas, casos_por_id, casos, not args.sin_juez)

    infos = bruto["infos"]
    resumenes = {b: resumir([f for f in filas if f["backend"] == b], infos[b]) for b in infos}
    # El contrato es de la cuantización: sólo tiene sentido si está la línea base fp32_cpu.
    # En una ablación (react vs. determinista) no se evalúa, y la tabla lo deja en blanco.
    hay_contrato = args.base in resumenes
    base_key = args.base if hay_contrato else next(iter(resumenes))
    # La línea base es el "antes": no se juzga contra sí misma (la cláusula de coste pide
    # pesar menos que la referencia, y ninguna referencia pesa menos que ella misma).
    veredictos = ({b: evaluar_contrato(r, resumenes[base_key])
                   for b, r in resumenes.items() if b != base_key}
                  if hay_contrato else {})
    if not hay_contrato:
        print(f"(sin la línea base '{args.base}' no se juzga el contrato; "
              f"se compara contra '{base_key}')")

    salida = {**{k: v for k, v in bruto.items() if k not in ("filas", "casos")},
              "linea_base": base_key, "contrato": CONTRATO,
              "variantes": resumenes, "veredictos": veredictos,
              "con_juez": not args.sin_juez}

    destino = Path(str(ruta).replace("_crudo.json", ".json"))
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=1)

    campos = [k for k in filas[0] if k not in ("respuesta_modelo", "respuesta_final")]
    campos += ["respuesta_modelo", "respuesta_final"]
    with open(str(destino).replace(".json", ".csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for fila in filas:
            w.writerow({k: fila.get(k) for k in campos})

    md = tabla_markdown(salida)
    with open(str(destino).replace(".json", ".md"), "w", encoding="utf-8") as f:
        f.write(f"# Cuantización del redactor — tabla antes/después ({salida['etiqueta']})\n\n")
        f.write(md + "\n\n## Veredicto del contrato\n\n")
        for b, v in veredictos.items():
            f.write(f"\n### `{b}` — {'CUMPLE' if v['cumple_contrato'] else 'NO CUMPLE'}\n\n")
            f.write("| cláusula | regla | obtenido | |\n|---|---|---|---|\n")
            for k, d in v["detalle"].items():
                f.write(f"| `{k}` | {d['regla']} | {d['obtenido']} | "
                        f"{'✅' if d['cumple'] else '❌'} |\n")
    print("\n" + md)
    print(f"\nEscrito: {destino} / .csv / .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
