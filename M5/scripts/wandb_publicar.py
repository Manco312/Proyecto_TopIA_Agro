"""Sube a W&B lo medido en M5: la tabla de cuantización, las dos ablaciones y el log de
peticiones del servicio.

Sin `WANDB_API_KEY` corre en modo **offline** y deja la corrida en `wandb/` para sincronizar
después con `wandb sync`. Así el dashboard se puede construir sin cuenta y publicar luego.

Uso:
    python scripts/wandb_publicar.py
    WANDB_API_KEY=... python scripts/wandb_publicar.py --online
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agro import config                                                   # noqa: E402


def _leer(p):
    if not Path(p).exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--proyecto", default=config.WANDB_PROYECTO)
    args = ap.parse_args()

    if not (args.online and os.environ.get("WANDB_API_KEY")):
        os.environ["WANDB_MODE"] = "offline"
        print("Modo offline (sin WANDB_API_KEY). Sincroniza después con: wandb sync wandb/<run>")

    import wandb

    R = config.DIR_RESULTADOS
    efi_cpu = _leer(R / "eficiencia_cpu.json")
    efi_gpu = _leer(R / "eficiencia_gpu.json")
    abl_plan = _leer(R / "ablacion_planificador.json")
    abl_filtro = (_leer(R / "ablacion_filtro_cpu.json")
                  or _leer(R / "ablacion_filtro_gpu.json"))

    run = wandb.init(project=args.proyecto, job_type="reporte-m5",
                     name=f"m5-{config.VERSION}",
                     config={"version": config.VERSION,
                             "modelo": config.MODEL_BASE_ID,
                             "lora": Path(config.MODELO_LORA).name,
                             "backend_desplegado": "int8_cpu",
                             "planificador_desplegado": "determinista"})

    # --- 1. Tabla de cuantización (el "antes/después" con su contrato) ---------------
    for etiqueta, datos in (("cpu", efi_cpu), ("gpu", efi_gpu)):
        if not datos:
            continue
        t = wandb.Table(columns=["variante", "dispositivo", "precision", "mib_pesos",
                                 "p50_gen_s", "p95_gen_s", "tokens_por_s", "sim_gold",
                                 "aciertos_gold", "juez_medio", "dosis_en_salida",
                                 "cumple_contrato"])
        for b, r in datos["variantes"].items():
            lat = r["latencia_generacion_s"]
            v = datos["veredictos"].get(b, {})
            t.add_data(b, r["dispositivo"], str(r["dtype"]), r["mib_pesos"], lat["p50"],
                       lat["p95"], r["tokens_por_s_media"], r["calidad"]["sim_media_gold"],
                       r["calidad"]["aciertos_gold"], r["calidad"]["juez_medio"],
                       r["seguridad"]["dosis_en_la_salida"],
                       bool(v.get("cumple_contrato")))
            wandb.log({f"eficiencia_{etiqueta}/{b}/mib_pesos": r["mib_pesos"],
                       f"eficiencia_{etiqueta}/{b}/p50_s": lat["p50"],
                       f"eficiencia_{etiqueta}/{b}/p95_s": lat["p95"],
                       f"eficiencia_{etiqueta}/{b}/tokens_por_s": r["tokens_por_s_media"],
                       f"eficiencia_{etiqueta}/{b}/sim_gold": r["calidad"]["sim_media_gold"],
                       f"eficiencia_{etiqueta}/{b}/juez": r["calidad"]["juez_medio"]})
        wandb.log({f"tabla/eficiencia_{etiqueta}": t})

    # --- 2. Ablaciones ---------------------------------------------------------------
    if abl_plan:
        t = wandb.Table(columns=["ruta", "p50_s", "p95_s", "sim_gold", "aciertos_gold",
                                 "abstencion_adv", "juez_medio"])
        for b, r in abl_plan["variantes"].items():
            lat = r["latencia_generacion_s"]
            t.add_data(b, lat["p50"], lat["p95"], r["calidad"]["sim_media_gold"],
                       r["calidad"]["aciertos_gold"], r["calidad"]["abstencion_adv"],
                       r["calidad"]["juez_medio"])
        wandb.log({"tabla/ablacion_planificador": t})

    if abl_filtro:
        t = wandb.Table(columns=["metrica", "sin_filtro", "con_filtro"])
        for k in abl_filtro["sin_filtro"]:
            t.add_data(k, abl_filtro["sin_filtro"][k], abl_filtro["con_filtro"][k])
        wandb.log({"tabla/ablacion_filtro_salida": t})

    # --- 3. El log de producción -----------------------------------------------------
    if Path(config.LOG_JSONL).exists():
        cols = ["ts", "ruta", "backend", "latencia_total_s", "latencia_vision_s",
                "p_top1", "valvula", "filtro_n", "tokens"]
        t = wandb.Table(columns=cols)
        n = 0
        with open(config.LOG_JSONL, encoding="utf-8") as f:
            for linea in f:
                try:
                    d = json.loads(linea)
                except json.JSONDecodeError:
                    continue
                t.add_data(d.get("ts"), d.get("ruta"), d.get("backend"),
                           d["latencia_s"].get("total"), d["latencia_s"].get("vision"),
                           d["confianza"].get("p_top1"), d["valvula"].get("disparo"),
                           d["filtro_salida"].get("n_frases_retiradas"),
                           d.get("tokens_generados"))
                n += 1
        wandb.log({"tabla/peticiones": t, "produccion/n_peticiones": n})
        print(f"   {n} peticiones del log subidas")

    # --- 4. Artefactos ---------------------------------------------------------------
    art = wandb.Artifact("m5-resultados", type="resultados")
    for p in sorted(R.glob("*.json")) + sorted(R.glob("*.csv")) + sorted(R.glob("*.md")):
        art.add_file(str(p))
    manifiesto = config.DIR_INDICE / "manifiesto.json"
    if manifiesto.exists():
        art.add_file(str(manifiesto))
    run.log_artifact(art)

    url = getattr(run, "url", None)
    wandb.finish()
    print(f"\nListo. {'Dashboard: ' + url if url else 'Corrida offline en wandb/'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
