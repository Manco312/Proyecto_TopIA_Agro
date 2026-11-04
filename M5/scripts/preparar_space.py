"""Arma la carpeta que se sube al Space de Hugging Face.

El repo del curso y el repo del Space no son la misma cosa: el Space sólo necesita lo que
hace falta para responder (app, paquete, artefactos, adaptador LoRA y ejemplos), no los
notebooks, los scripts de medición ni los resultados. Este script copia exactamente eso y
escribe el `README.md` con el *front-matter* que Spaces espera.

Uso:
    python scripts/preparar_space.py
    python scripts/preparar_space.py --destino ../space --con-ejemplos
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agro import config                                                   # noqa: E402

FRONT_MATTER = """---
title: Recomendador de manejo agronomico
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: {sdk}
app_file: app.py
pinned: false
license: mit
suggested_hardware: zero-a10g
short_description: Foto de una hoja + pregunta, y una recomendacion con su ficha tecnica.
---
"""

CUERPO = """
# 🌿 Recomendador de manejo agronómico

**"Esta hoja de mi cultivo se ve así — ¿qué tiene y qué le hago?"**

Sistema completo del curso *Tópicos Especiales y Aplicaciones en IA* (EAFIT), módulos M1–M5.
Equipo: Luciana Hoyos · Sara López · Juan Carlos Citelly · Santiago Manco Maya.

| Pieza | Qué es |
|---|---|
| Clasificador visual | CLIP ViT-B/32 congelado + cabeza lineal de 38 clases de PlantVillage (91.4 % top-1) |
| Válvula | 4 guardas calibradas al percentil 5 del retenido: no diagnostica cuando no debe |
| Recuperación | 151 pasajes de 50 fichas · BM25 + denso (RRF) + reranker cross-encoder |
| Redactor | Qwen2.5-0.5B + adaptador LoRA propio, fp16 en **ZeroGPU** |
| Filtro de salida | hoja sana → sin receta · sin diagnóstico → sin receta · **nunca una dosis** |

## Arquitectura

{diagrama}

> ⚠️ Recomienda manejo agronómico; **no sustituye a un agrónomo ni al ICA**. El clasificador
> se entrenó con PlantVillage (hojas arrancadas sobre fondo uniforme): ante una foto de campo
> la válvula puede dispararse y dejarte sin diagnóstico. Nunca entrega dosis ni periodos de
> carencia: esos se leen en la etiqueta del producto registrado ante el ICA.

La pestaña **Estado** es el health check; la pestaña **Cómo funciona** explica las rutas y
el filtro. El código, las mediciones y el reporte final están en el repo del curso.

## Hardware

Este Space corre en **ZeroGPU**: la GPU se asigna sólo mientras se ejecuta la función
decorada con `@spaces.GPU` (`_responder_en_gpu` en `app.py`), con un tope de 120 s por
petición. Los modelos se cargan al arrancar y se mandan a `cuda` ahí mismo — es el patrón
documentado de ZeroGPU, y la reserva real se difiere hasta la primera petición.

El logging, los contadores y el health check se quedan **fuera** de la función decorada: la
función corre en otro proceso y lo que mute dentro no vuelve.

## Secretos (Settings → Variables and secrets)

| Nombre | Para qué | ¿Obligatorio? |
|---|---|---|
| `WANDB_API_KEY` | observabilidad en W&B | no |
| `AGRO_WANDB` | ponlo en `1` para activar W&B | no |
| `GROQ_API_KEY` | backend opcional del redactor | no |

Sin ninguno, el Space funciona: usa el modelo propio del curso y registra en el log local.
"""

# Lo que el Space necesita y nada más.
COPIA_DIRECTORIOS = ["agro", "artefactos"]
COPIA_ARCHIVOS = ["app.py", "requirements.txt", ".env.example"]
EXCLUIR = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints")


def _diagrama():
    """El diagrama vive en el README del repo; aquí se reutiliza para no mantener dos."""
    try:
        texto = (config.RAIZ / "README.md").read_text(encoding="utf-8")
        return "```mermaid" + texto.split("```mermaid")[1].split("```")[0] + "```"
    except Exception:
        return "_(ver el README del repo)_"


def _partir_front_matter(texto):
    """Devuelve (front_matter, cuerpo). El front-matter es lo que configura el Space."""
    if texto.startswith("---"):
        lineas = texto.splitlines(keepends=True)
        for i, l in enumerate(lineas[1:], 1):
            if l.strip() == "---":
                return "".join(lineas[:i + 1]), "".join(lineas[i + 1:])
    return "", texto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", default=str(config.RAIZ / "build_space"))
    ap.add_argument("--lora", default=config.MODELO_LORA)
    ap.add_argument("--con-ejemplos", action="store_true",
                    help="copiar las 7 imágenes del eval set de M4 como ejemplos de la app")
    ap.add_argument("--sdk", default=None, help="sdk_version de Gradio (por defecto, la local)")
    ap.add_argument("--rehacer-readme", action="store_true",
                    help="sobrescribir el README.md del Space (pisa el front-matter)")
    args = ap.parse_args()

    destino = Path(args.destino)
    # NO se borra la carpeta entera: en cuanto se hace `git init` ahí vive el .git del
    # Space, y con él el historial y el remoto. Se reemplazan sólo las rutas que este
    # script gestiona.
    hay_git = (destino / ".git").exists()
    # Un vision_config.json editado en el destino es una decisión deliberada (mover el
    # punto de operación de la válvula). No se pisa: se avisa y se conserva.
    cfg_destino = destino / "artefactos" / "vision" / "vision_config.json"
    cfg_previo = cfg_destino.read_bytes() if cfg_destino.exists() else None
    for rel in COPIA_DIRECTORIOS + ["mi-modelo-lora", "ejemplos"]:
        p = destino / rel
        if p.is_dir():
            shutil.rmtree(p)
    destino.mkdir(parents=True, exist_ok=True)

    for d in COPIA_DIRECTORIOS:
        shutil.copytree(config.RAIZ / d, destino / d, ignore=EXCLUIR)
    for f in COPIA_ARCHIVOS:
        shutil.copy2(config.RAIZ / f, destino / f)

    if cfg_previo is not None and cfg_previo != cfg_destino.read_bytes():
        cfg_destino.write_bytes(cfg_previo)
        print("   artefactos/vision/vision_config.json: se conserva el del destino "
              "(estaba modificado). Usa AGRO_UMBRAL_PROTO para moverlo sin tocarlo.")

    lora = Path(args.lora)
    assert lora.is_dir(), f"no encuentro el adaptador LoRA en {lora}"
    shutil.copytree(lora, destino / "mi-modelo-lora", ignore=EXCLUIR)

    if args.con_ejemplos:
        origen_img = config.ORIGEN_DATOS / "eval_set" / "imagenes"
        csv_casos = config.ORIGEN_DATOS / "eval_set" / "casos_m4.csv"
        if origen_img.is_dir() and csv_casos.exists():
            import csv as _csv
            (destino / "ejemplos").mkdir()
            manifiesto = []
            with open(csv_casos, encoding="utf-8", newline="") as f:
                for fila in _csv.DictReader(f):
                    p = origen_img / fila["imagen"]
                    if p.exists():
                        shutil.copy2(p, destino / "ejemplos" / fila["imagen"])
                        manifiesto.append({"imagen": fila["imagen"],
                                           "pregunta": fila["pregunta"], "id": fila["id"]})
            with open(destino / "ejemplos" / "ejemplos.json", "w", encoding="utf-8") as f:
                json.dump(manifiesto, f, ensure_ascii=False, indent=1)
            print(f"   ejemplos: {len(manifiesto)} imágenes del eval set de M4")

    try:
        import gradio
        sdk = args.sdk or gradio.__version__
    except Exception:
        sdk = args.sdk or "6.31.0"
    readme = destino / "README.md"
    cuerpo = CUERPO.format(diagrama=_diagrama())
    # El front-matter configura el Space (SDK, versión, hardware) y puede estar ajustado a
    # mano: perderlo rompe el build de una forma que no se ve hasta que falla. Se conserva
    # el que haya y se refresca sólo el cuerpo.
    if readme.exists() and not args.rehacer_readme:
        fm, _ = _partir_front_matter(readme.read_text(encoding="utf-8"))
        readme.write_text((fm or FRONT_MATTER.format(sdk=sdk)) + cuerpo, encoding="utf-8")
        print("   README.md: front-matter conservado, cuerpo actualizado")
    else:
        readme.write_text(FRONT_MATTER.format(sdk=sdk) + cuerpo, encoding="utf-8")
    (destino / ".gitignore").write_text(
        "\n".join(["__pycache__/", "*.pyc", "logs/", "wandb/", ".env", ""]),
        encoding="utf-8")
    (destino / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
        "*.npy filter=lfs diff=lfs merge=lfs -text\n"
        "*.joblib filter=lfs diff=lfs merge=lfs -text\n"
        "tokenizer.json filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")

    publicados = [p for p in destino.rglob("*")
                  if p.is_file() and ".git" not in p.relative_to(destino).parts]
    total = sum(p.stat().st_size for p in publicados)
    n = len(publicados)
    print(f"\nCarpeta del Space lista: {destino}")
    print(f"{n} archivos · {total / 2**20:.1f} MB · gradio sdk_version={sdk}\n")
    print("Para subirla:\n"
          f"  cd {destino}\n"
          "  git init && git lfs install\n"
          "  git remote add origin https://huggingface.co/spaces/<usuario>/<space>\n"
          "  git add -A && git commit -m 'M5: sistema completo' && git push -u origin main\n"
          "o con la CLI:\n"
          f"  hf upload <usuario>/<space> {destino} . --repo-type=space\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
