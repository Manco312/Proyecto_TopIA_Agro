"""Ablación 1 — ¿qué aporta el filtro de salida de M5?

No genera nada: reutiliza las respuestas del modelo que ya produjo `bench_eficiencia.py` y
recompone la respuesta final **con** y **sin** filtro. Al partir de los mismos tokens, la
comparación es exacta: cualquier diferencia es del filtro y de nada más.

Uso:
    python scripts/ablacion_filtro.py resultados/eficiencia_cpu_crudo.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agro.catalogo import catalogo                                        # noqa: E402
from agro.sistema import filtro_salida                                    # noqa: E402
from agro.texto import dio_dosis_numerica, dio_prescripcion               # noqa: E402


def recomponer(fila, prep, con_filtro):
    """Rehace la respuesta final a partir del texto del modelo, con o sin filtro. El prefijo
    determinista (encabezado de la foto o aviso de válvula) quedó grabado en la pasada 1."""
    texto = fila["respuesta_modelo"] or ""
    filtrado, informe = filtro_salida(texto, fila["ruta"], prep.get("etiqueta_diag"),
                                      activo=con_filtro)
    return prep.get("prefijo", "") + filtrado, informe


def contar(filas, preparacion, con_filtro):
    por_id = {p["caso_id"]: p for p in preparacion}
    n_dosis = n_receta_sana = n_tocadas = 0
    tocados = []
    for f in filas:
        resp, informe = recomponer(f, por_id.get(f["id"], {}), con_filtro)
        etq = por_id.get(f["id"], {}).get("etiqueta_diag")
        sana = bool(etq and catalogo().get(etq, {}).get("tipo") == "sano")
        dosis = dio_dosis_numerica(resp)
        receta_sana = sana and dio_prescripcion(resp)
        n_dosis += dosis
        n_receta_sana += receta_sana
        if informe["n_frases_retiradas"]:
            n_tocadas += 1
            tocados.append({"backend": f["backend"], "id": f["id"],
                            "reglas": informe["reglas_aplicadas"],
                            "retirado": informe["frases_retiradas"]})
    return {"respuestas_con_dosis": n_dosis,
            "respuestas_que_recetan_sobre_hoja_sana": n_receta_sana,
            "respuestas_modificadas": n_tocadas}, tocados


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("crudo")
    args = ap.parse_args()

    with open(args.crudo, encoding="utf-8") as f:
        bruto = json.load(f)
    filas, prep = bruto["filas"], bruto["preparacion"]

    sin, _ = contar(filas, prep, con_filtro=False)
    con, tocados = contar(filas, prep, con_filtro=True)

    salida = {"fuente": Path(args.crudo).name, "n_respuestas": len(filas),
              "variantes_del_redactor": sorted({f["backend"] for f in filas}),
              "sin_filtro": sin, "con_filtro": con, "casos_tocados": tocados}

    destino = Path(args.crudo).parent / f"ablacion_filtro_{bruto['etiqueta']}.json"
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=1)

    print(f"Ablación del filtro de salida sobre {len(filas)} respuestas "
          f"({', '.join(salida['variantes_del_redactor'])})\n")
    print("| | sin filtro | con filtro |")
    print("|---|---:|---:|")
    for k in sin:
        print(f"| {k.replace('_', ' ')} | {sin[k]} | {con[k]} |")
    if tocados:
        print("\nCasos en los que el filtro cambió la respuesta:")
        for t in tocados:
            print(f"  - [{t['backend']}] {t['id']} · {', '.join(t['reglas'])}")
            for frase in t["retirado"]:
                print(f"      retirado: {frase.strip()[:150]}")
    print(f"\nEscrito: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
