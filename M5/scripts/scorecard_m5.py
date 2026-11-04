"""El scorecard de M5 junto a los de M3 y M4, con las mismas métricas y el mismo eval set.

Lee los resultados ya puntuados de una variante (por defecto la que se despliega) y los pone
al lado de los números que M3 y M4 dejaron escritos en sus propios scorecards. Las métricas
son las mismas funciones de M2 (`acierto_dominio`, juez por valor esperado), y el conjunto de
casos es el mismo: 27 de texto + 7 con imagen.

La única diferencia declarada: **la Dimensión 1 se calcula sobre el texto del modelo**, no
sobre la respuesta con el encabezado determinista encima. Es la corrección que M4 §9 dejó
pedida al medir que ese encabezado baja la similitud entre 0.018 y 0.242.

Uso:
    python scripts/scorecard_m5.py
    python scripts/scorecard_m5.py --variante int8_cpu --resultados resultados/eficiencia_cpu.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agro import config                                                   # noqa: E402


def _f(x, d=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def _b(x):
    return str(x).strip().lower() in ("true", "1")


def resumen_grupo(filas):
    gold = [f for f in filas if f["tipo"] == "gold"]
    adv = [f for f in filas if f["tipo"] == "adversarial"]
    sims = [_f(f["sim"]) for f in gold if _f(f["sim"]) is not None]
    jueces = [_f(f["juez"]) for f in filas if _f(f["juez"]) is not None]
    lat = [_f(f["segundos_generacion"], 0.0) for f in filas if not _b(f["sin_llm"])]
    return {
        "n": len(filas),
        "gold_aciertos_dominio": f"{sum(_b(f['acierto']) for f in gold)}/{len(gold)}",
        "gold_sim_prom": round(stats.mean(sims), 3) if sims else None,
        "gold_juez_prom": round(stats.mean(jueces), 3) if jueces else None,
        "adversariales_se_abstuvo": f"{sum(_b(f['abstuvo']) for f in adv)}/{len(adv)}",
        "latencia_generacion_prom_s": round(stats.mean(lat), 2) if lat else None,
        "valvula_total": sum(1 for f in filas if f["ruta"] == "valvula"),
        # Mismo numerador que M4: válvula disparada SOBRE LOS ADVERSARIALES con foto.
        "valvula_adv": f"{sum(1 for f in adv if f['ruta'] == 'valvula')}/{len(adv)}",
        "filtro_activado": sum(1 for f in filas if int(_f(f["filtro_n"], 0))),
        "dosis_en_la_salida": sum(1 for f in filas if _b(f["dosis_final"])),
        "receta_sobre_hoja_sana": sum(1 for f in filas if _b(f["receta_sobre_sana"])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resultados", default=str(config.DIR_RESULTADOS / "eficiencia_cpu.json"))
    ap.add_argument("--variante", default="int8_cpu")
    args = ap.parse_args()

    res_json = Path(args.resultados)
    res_csv = Path(str(res_json).replace(".json", ".csv"))
    assert res_csv.exists(), f"no encuentro {res_csv}; corre antes scripts/puntuar.py"

    with open(res_csv, encoding="utf-8", newline="") as f:
        filas = [r for r in csv.DictReader(f) if r["backend"] == args.variante]
    assert filas, f"no hay filas de la variante {args.variante} en {res_csv}"

    texto = [f for f in filas if f["fuente"].startswith("eval_set")]
    imagen = [f for f in filas if f["fuente"] == "imagen_m4"]

    m5 = {"variante": args.variante, "texto": resumen_grupo(texto),
          "imagen": resumen_grupo(imagen), "todo": resumen_grupo(filas)}

    # Los números que M3 y M4 dejaron escritos, leídos de sus propios artefactos.
    ref = {}
    p3 = config.ORIGEN_DATOS / "scorecard_rag_m3.json"
    p4 = config.ORIGEN_DATOS / "scorecard_m4.json"
    if p3.exists():
        with open(p3, encoding="utf-8") as f:
            a = json.load(f)["3_rag_agentico"]
        ref["M3 · rag agéntico"] = {
            "gold_aciertos_dominio": a["gold"]["aciertos_dominio"],
            "gold_sim_prom": a["gold"]["sim_embeddings_prom"],
            "gold_juez_prom": a["gold"]["llm_juez_prom"],
            "adversariales_se_abstuvo": a["adversarial"]["se_abstuvo"],
            "latencia_prom_s": a["latencia"]["seg_por_consulta_prom"]}
    if p4.exists():
        with open(p4, encoding="utf-8") as f:
            b = json.load(f)
        t = b["texto_m3"]
        ref["M4 · + visión"] = {
            "gold_aciertos_dominio": t["gold_aciertos_dominio"],
            "gold_sim_prom": t["gold_sim_prom"], "gold_juez_prom": t["gold_juez_prom"],
            "adversariales_se_abstuvo": t["adversariales_se_abstuvo"],
            "latencia_prom_s": t["latencia_prom_s"]}
        ref["_m4_con_imagen"] = b["con_imagen"]

    salida = {"variante_desplegada": args.variante, "m5": m5, "referencias": ref,
              "nota_dimension_1": "sim calculada sobre respuesta_modelo (corrección de M4 §9)"}
    destino = config.DIR_RESULTADOS / "scorecard_m5.json"
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=1)

    t = m5["texto"]
    print("## 27 casos de texto — mismas métricas, mismo eval set\n")
    etiqueta_m5 = f"M5 · `{args.variante}`"
    print(f"| | M3 · rag agéntico | M4 · + visión | {etiqueta_m5} |")
    print("|---|---|---|---|")
    claves = [("gold · aciertos de dominio", "gold_aciertos_dominio"),
              ("gold · similitud", "gold_sim_prom"),
              ("juez (1–5)", "gold_juez_prom"),
              ("adversariales · se abstuvo", "adversariales_se_abstuvo")]
    for etiqueta, k in claves:
        a = ref.get("M3 · rag agéntico", {}).get(k, "—")
        b = ref.get("M4 · + visión", {}).get(k, "—")
        v = t.get(k)
        if k == "gold_juez_prom" and not v:
            v = "no medido en esta corrida"
        print(f"| {etiqueta} | {a} | {b} | {v} |")
    print(f"| latencia media | {ref.get('M3 · rag agéntico', {}).get('latencia_prom_s', '—')} s "
          f"| {ref.get('M4 · + visión', {}).get('latencia_prom_s', '—')} s "
          f"| {t['latencia_generacion_prom_s']} s (sólo generación, 2 vCPU) |")

    i = m5["imagen"]
    cim = ref.get("_m4_con_imagen", {})
    print("\n## 7 casos con imagen\n")
    print("| | M4 | M5 |")
    print("|---|---|---|")
    print(f"| gold · aciertos de dominio | {cim.get('gold_aciertos_dominio', '—')} "
          f"| {i['gold_aciertos_dominio']} |")
    print(f"| adversariales · válvula | {cim.get('adversariales_valvula', '—')} "
          f"| {i['valvula_adv']} |")
    print(f"| **respuestas con dosis** | no medido en M4 (`img-05` dio una) "
          f"| **{i['dosis_en_la_salida']}** |")
    print(f"| **recetas sobre hoja sana** | 1 (`img-05`) | **{i['receta_sobre_hoja_sana']}** |")
    print(f"| filtro de salida activado | — | {i['filtro_activado']} |")
    print(f"\nEscrito: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
