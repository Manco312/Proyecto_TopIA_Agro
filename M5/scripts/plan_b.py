"""El plan B de la demo: las respuestas ya corridas, en un documento que se puede proyectar.

La guía de despliegue del curso lo pide explícitamente — *"Plan B rotulado: video o capturas
de los casos ensayados, por si falla la red o la GPU"*. Esto lo genera del propio banco, así
que lo que se proyecta es **exactamente** lo que midió el scorecard, no una captura suelta.

Uso:
    python scripts/plan_b.py
    python scripts/plan_b.py --variante fp16_gpu --crudo resultados/eficiencia_gpu_crudo.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agro import config                                                   # noqa: E402

CABECERA = """# Plan B — casos ensayados

> Generado desde `{crudo}` con la variante `{variante}`. Son las respuestas **reales** del
> sistema sobre los casos del eval set, las mismas que puntúa el scorecard. Si el día de la
> demo falla la red o el Space está dormido, esto es lo que se proyecta.
>
> Entorno de la corrida: {entorno}

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crudo", default=str(config.DIR_RESULTADOS / "eficiencia_cpu_crudo.json"))
    ap.add_argument("--variante", default="fp32_cpu")
    ap.add_argument("--solo-imagen", action="store_true", default=True)
    ap.add_argument("--todos", dest="solo_imagen", action="store_false")
    args = ap.parse_args()

    with open(args.crudo, encoding="utf-8") as f:
        d = json.load(f)
    prep = {p["caso_id"]: p for p in d["preparacion"]}
    casos = {c["id"]: c for c in d["casos"]}
    filas = [x for x in d["filas"] if x["backend"] == args.variante]
    if args.solo_imagen:
        filas = [x for x in filas if x["fuente"] == "imagen_m4"]
    assert filas, f"no hay filas de {args.variante} en {args.crudo}"

    e = d["entorno"]
    partes = [CABECERA.format(
        crudo=Path(args.crudo).name, variante=args.variante,
        entorno=f"{e.get('sistema')} · {e.get('cpus')} CPU "
                f"(torch limitado a {e.get('hilos_torch')} hilos) · torch {e.get('torch')}")]

    for x in filas:
        p, c = prep.get(x["id"], {}), casos.get(x["id"], {})
        partes.append(f"## `{x['id']}`\n")
        if p.get("imagen"):
            partes.append(f"**Imagen:** `{Path(p['imagen']).name}`  \n")
        partes.append(f"**Pregunta:** {c.get('input', '')}\n\n")
        filas_tabla = [f"| ruta | `{x['ruta']}` |"]
        if p.get("etiqueta_pred"):
            filas_tabla.append(f"| etiqueta de la foto | `{p['etiqueta_pred']}` "
                               f"({(p.get('p_top1') or 0):.0%}) |")
        if p.get("valvula"):
            filas_tabla.append(f"| **válvula** | {p.get('motivos')} |")
        if x["filtro_n"]:
            filas_tabla.append(f"| **filtro de salida** | {x['filtro_n']} frase(s) · "
                               f"{x['filtro_reglas']} |")
        filas_tabla.append(f"| latencia de generación | {x['segundos_generacion']} s "
                           f"({x['tokens']} tokens) |")
        partes.append("| | |\n|---|---|\n" + "\n".join(filas_tabla) + "\n\n")
        partes.append("**Respuesta del sistema:**\n\n```\n"
                      + (x["respuesta_final"] or "").strip() + "\n```\n\n")
        if x["filtro_n"]:
            partes.append("<details><summary>Lo que el modelo había escrito antes del "
                          "filtro</summary>\n\n```\n"
                          + (x["respuesta_modelo"] or "").strip() + "\n```\n</details>\n\n")
        partes.append("---\n\n")

    destino = config.DIR_RESULTADOS / "plan_b_casos.md"
    destino.write_text("".join(partes), encoding="utf-8")
    print(f"{len(filas)} casos escritos en {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
