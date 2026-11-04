"""El Space: una app delgada sobre `agro.sistema.responder`.

Todo lo pesado pasa ANTES de la primera consulta: modelos cargados una vez, índice leído
de disco ya construido. La app no entrena, no indexa y no decide nada; sólo recibe la
pregunta y la foto, llama al sistema y escribe la línea de log.

Tres pestañas: la consulta, el estado (health check) y qué hace el sistema por dentro.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

try:                                   # en el Space (ZeroGPU). Debe importarse ANTES de torch.
    import spaces
    EN_ZEROGPU = True
except Exception:                      # no instalado, o instalado pero sin el entorno de
    # ZeroGPU (el paquete lanza AssertionError si SPACES_ZERO_GPU=true y falta el resto de
    # la configuración). En cualquiera de los dos casos: decorador que no hace nada.
    EN_ZEROGPU = False

    class _SinSpaces:
        @staticmethod
        def GPU(fn=None, **_kw):
            return fn if fn is not None else (lambda f: f)

    spaces = _SinSpaces()

import gradio as gr

from agro import config, observabilidad
from agro.catalogo import cultivos_cubiertos, detectar_cultivos
from agro.sistema import responder

RAIZ = Path(__file__).resolve().parent
EJEMPLOS_DIR = Path(os.environ.get("AGRO_EJEMPLOS", RAIZ / "ejemplos"))
# ZeroGPU asigna la GPU sólo mientras corre una función decorada, y con un tope de tiempo.
# 120 s sobra: la ruta más cara (ReAct, hasta 4 pasos del planificador + la redacción) va
# muy por debajo en una H200.
DURACION_GPU = int(os.environ.get("AGRO_GPU_DURACION", "120"))
ARRANQUE = {}


def _tag_git():
    """La versión con tag de Git visible en la app (cuidado de despliegue de la S15)."""
    for cmd in (["git", "describe", "--tags", "--always", "--dirty"],
                ["git", "rev-parse", "--short", "HEAD"]):
        try:
            r = subprocess.run(cmd, cwd=RAIZ, capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception:
            pass
    return os.environ.get("SPACE_COMMIT_SHA", "(sin git)")[:12] or "(sin git)"


TAG = _tag_git()

# Se añade DESPUÉS de `responder()`, no dentro: así el texto que miden los scorecards es
# exactamente el mismo que se midió en M2-M4, sin un pie de página encima que le baje la
# similitud (la lección de M4 §9 con el encabezado de la foto).
NOTA_EXTERNA = (
    "\n\n<sub>Redacción generada por un modelo externo (`{modelo}` vía Groq) "
    "<b>sobre las fichas de este sistema</b>. El diagnóstico lo da el clasificador propio "
    "de M4 y la evidencia, el índice de M3.</sub>")

AVISO = (
    "Este sistema **recomienda manejo agronómico; no sustituye a un agrónomo ni al ICA**. "
    "El diagnóstico lo da un clasificador de imágenes entrenado con PlantVillage (hojas "
    "arrancadas sobre fondo uniforme), no un fitopatólogo. **Nunca entrega dosis ni periodos "
    "de carencia**: esos se leen en la etiqueta del producto registrado ante el ICA."
)


def calentar():
    """Cargar una vez, atender muchas. Sin esto, el primer agricultor paga la carga.

    En ZeroGPU esto corre **sin GPU asignada**, y es correcto: es el patrón documentado.
    Los modelos se construyen y se mandan a `cuda` aquí; ZeroGPU difiere la reserva real
    hasta que una función decorada con `@spaces.GPU` se ejecuta. Lo que NO puede pasar aquí
    es una inferencia.
    """
    t0 = time.time()
    from agro import indice, vision
    from agro.modelos import fijar_semillas, redactor
    fijar_semillas()
    indice.cargar()
    vision.cargar()
    red = redactor()
    if config.resolver_ruta() == "react" and config.resolver_motor_planificador() == "local":
        from agro.modelos import planificador
        planificador()      # con Groq de cerebro no hace falta cargar el 1.5B
    seg = time.time() - t0
    observabilidad.marcar_arranque(seg, ["indice", "vision", f"redactor:{red.backend}",
                                         f"planificador:{config.resolver_ruta()}"])
    observabilidad.iniciar_wandb()
    ARRANQUE.update({"segundos": round(seg, 1), "redactor": red.info()})
    print(json.dumps({"evento": "arranque", "version": config.VERSION, "tag": TAG,
                      "segundos": round(seg, 1), "redactor": red.info()}, ensure_ascii=False))
    return ARRANQUE


def _resumen(d):
    """Lo que el usuario puede querer ver sin leer un JSON."""
    filas = [f"| ruta | `{d['ruta']}` |",
             f"| backend del redactor | `{d['backend']}` |",
             f"| planificador | `{d['planificador']}` |",
             f"| latencia total | {d['latencia_total_s']} s |"]
    if d["etiqueta_pred"]:
        filas.insert(1, f"| etiqueta de la foto | `{d['etiqueta_pred']}` ({d['p_top1']:.0%}) |")
        filas.insert(2, f"| guardas | p_hoja={d['p_hoja']} · cos_prototipo={d['cos_prototipo']} |")
        filas.append(f"| latencia del componente visual | {d['latencia_vision_s']} s |")
    if d["valvula"]:
        filas.append(f"| **válvula visual** | {d['motivos']} |")
    f = d["filtro"] or {}
    if f.get("n_frases_retiradas"):
        filas.append(f"| **filtro de salida** | {f['n_frases_retiradas']} frase(s): "
                     f"{', '.join(f['reglas_aplicadas'])} |")
    return "| | |\n|---|---|\n" + "\n".join(filas)

def _necesita_gpu():
    """¿Hay algún modelo nuestro que de verdad quiera GPU en esta configuración?

    Con Groq redactando y eligiendo herramientas, lo único que queda del lado del equipo es
    CLIP (0,15 s) y el cross-encoder, que van de sobra en CPU. Reservar un *slot* de
    ZeroGPU para eso significaría tener la GPU ocupada **mientras se espera la respuesta
    HTTP de Groq**: se gasta cuota por no hacer nada. Así que en esa configuración se
    responde fuera del decorador.
    """
    return (config.resolver_backend() != "groq"
            or config.resolver_motor_planificador() == "local")


@spaces.GPU(duration=DURACION_GPU)
def _responder_en_gpu(pregunta, imagen):
    """Lo ÚNICO que necesita GPU. Devuelve datos planos, no objetos.

    ZeroGPU ejecuta la función decorada en un proceso aparte con la GPU asignada, así que
    todo lo que mute aquí dentro se pierde al volver. Por eso el logging, los contadores y
    el health check se quedan fuera: si `observabilidad.registrar()` se llamara aquí, la
    pestaña de Estado diría siempre "0 peticiones".
    """
    return responder(pregunta, imagen)


def consultar(imagen, pregunta, progress=gr.Progress()):
    pregunta = (pregunta or "").strip()
    if not pregunta and imagen is None:
        return ("Escribe tu pregunta (y, si puedes, adjunta una foto de la hoja).", "", "{}")
    if not pregunta:
        pregunta = "Que tiene esta hoja y que debo hacer?"

    progress(0.1, desc="leyendo la hoja…" if imagen is not None else "consultando las fichas…")
    detalle = {"con_foto": imagen is not None,
               "cultivos_detectados": detectar_cultivos(pregunta)}
    try:
        resp, d = (_responder_en_gpu(pregunta, imagen) if _necesita_gpu()
                   else responder(pregunta, imagen))
        detalle.update(d)
        observabilidad.registrar(pregunta, detalle)       # en el proceso principal
        # Se declara fuera de `responder()` a propósito: así el texto que mide el scorecard
        # es exactamente el mismo que se midió en M2-M4, sin un pie de página encima.
        if d.get("backend") == "groq":
            resp += NOTA_EXTERNA.format(modelo=config.GROQ_MODELO)
        return resp, _resumen(d), json.dumps(
            {k: v for k, v in d.items() if k not in ("respuesta_modelo", "traza")},
            ensure_ascii=False, indent=1)
    except Exception as e:                                                # noqa: BLE001
        observabilidad.registrar(pregunta, detalle, error=e)
        return (f"El sistema falló en esta consulta: `{type(e).__name__}`. "
                "Inténtalo de nuevo; si persiste, avisa al equipo.", "", "{}")


def estado():
    s = observabilidad.salud()
    s["tag_git"] = TAG
    s["zero_gpu"] = {"activo": EN_ZEROGPU, "duracion_max_s": DURACION_GPU,
                     "se_usa_en_cada_peticion": _necesita_gpu()}
    s["configuracion"]["motor_planificador"] = config.resolver_motor_planificador()
    s["configuracion"]["groq_key"] = bool(config.GROQ_API_KEY)
    s["configuracion"]["max_tokens_respuesta"] = (
        config.MAX_NEW_GROQ if config.resolver_backend() == "groq"
        else config.MAX_NEW_SISTEMA)
    s["configuracion"]["max_ctx_chars"] = config.MAX_CTX_CHARS
    return s


def ultimas_lineas(n=15):
    try:
        with open(config.LOG_JSONL, encoding="utf-8") as f:
            lineas = f.readlines()[-n:]
        return "".join(lineas) or "(aún no hay peticiones)"
    except OSError:
        return "(aún no hay peticiones)"


def _ejemplos():
    """Los 7 casos con imagen de M4. En el Space viven en `ejemplos/` (los copia
    scripts/preparar_space.py); corriendo desde el repo se leen del eval set de M4."""
    carpeta, manifiesto = EJEMPLOS_DIR, EJEMPLOS_DIR / "ejemplos.json"
    if manifiesto.exists():
        with open(manifiesto, encoding="utf-8") as f:
            return [[str(carpeta / e["imagen"]), e["pregunta"]] for e in json.load(f)
                    if (carpeta / e["imagen"]).exists()]

    import csv
    csv_casos = config.ORIGEN_DATOS / "eval_set" / "casos_m4.csv"
    imgs = config.ORIGEN_DATOS / "eval_set" / "imagenes"
    if csv_casos.exists() and imgs.is_dir():
        with open(csv_casos, encoding="utf-8", newline="") as f:
            return [[str(imgs / fila["imagen"]), fila["pregunta"]]
                    for fila in csv.DictReader(f) if (imgs / fila["imagen"]).exists()]
    if carpeta.is_dir():
        return [[str(p), "Que tiene esta hoja y que debo hacer?"]
                for p in sorted(carpeta.glob("*.jpg"))]
    return []


COMO_FUNCIONA = f"""
### Qué hay debajo

| Módulo | Pieza | Qué hace aquí |
|---|---|---|
| M1 | Qwen2.5-0.5B + LoRA | redacta la respuesta final |
| M2 | regla de dominio y rúbrica | define qué cuenta como acierto (en el bench, no en vivo) |
| M3 | 151 pasajes de 50 fichas · BM25 + denso (RRF) + reranker · 3 herramientas | trae la evidencia |
| M4 | CLIP ViT-B/32 + cabeza lineal de 38 clases · válvula de 4 guardas | lee la foto |
| M5 | cuantización int8 · filtro de salida · health check · logging · W&B | lo pone en producción |

### Las tres rutas

1. **Sin foto** → diagnóstico diferencial → ficha → redacción.
2. **Con foto y el clasificador seguro** → la etiqueta decide qué ficha se consulta. Es la ruta
   *más rápida*: se salta la búsqueda.
3. **Con foto dudosa** → la **válvula** se dispara (no es una hoja · confianza < {{p}} ·
   lejos de lo entrenado < {{c}} · la foto dice otro cultivo que tu pregunta) y el sistema
   vuelve a la ruta de texto, avisándote.

### El filtro de salida

Tres reglas deterministas sobre el texto del modelo, posteriores a la redacción:

1. **hoja sana → sin receta.** Si la foto dice "sana", ninguna frase puede recomendar aplicar
   un producto.
2. **sin diagnóstico → sin receta.** Si la válvula se disparó, tampoco.
3. **sin dosis, nunca.** Ninguna de las 50 fichas del corpus contiene una cifra de dosis, así
   que **toda** dosis numérica en la salida la inventó el modelo. Se retira y se redirige a la
   etiqueta del producto.

La regla 3 existe por un fallo concreto y medido: en M4, sobre una hoja de tomate **sana**, el
redactor escribió *"se recomienda fungicida foliar… entre 50-100 gramos por hectárea"* justo
debajo del encabezado que decía que no había nada que tratar.

### Cultivos cubiertos

{{cultivos}}

Cualquier otro cultivo: el sistema se abstiene y te manda al ICA. Es a propósito.

### Límites

- **PlantVillage no es el campo.** El 91.4 % de acierto del clasificador se midió sobre 760
  hojas arrancadas, con fondo uniforme y luz de laboratorio. Ante una foto de campo real la
  válvula puede dispararse y dejarte sin diagnóstico — es el límite más serio del sistema.
- El redactor es un modelo de **0.5B**: puede escribir frases vagas o mal construidas aunque
  la ficha recuperada sea la correcta. El cuello de botella está medido y documentado desde M3.
- Una sola corrida por caso, *greedy*, sin intervalos de confianza.
"""


def construir():
    from agro import vision
    v = vision.cargar()
    como = COMO_FUNCIONA.format(p=v["umbral_p"], c=v["umbral_proto"],
                                cultivos=", ".join(cultivos_cubiertos()))

    # En Gradio 6 el tema se pasa a launch(), no al constructor de Blocks.
    with gr.Blocks(title="Recomendador de manejo agronómico") as demo:
        gr.Markdown(
            f"# 🌿 Recomendador de manejo agronómico\n"
            f"**Esta hoja de mi cultivo se ve así — ¿qué tiene y qué le hago?**\n\n"
            f"<sub>Tópicos Especiales y Aplicaciones en IA · EAFIT · M5 · "
            f"versión `{config.VERSION}` · git `{TAG}`</sub>")
        gr.Markdown(f"> ⚠️ {AVISO}")

        with gr.Tab("Consulta"):
            with gr.Row():
                with gr.Column(scale=1):
                    img = gr.Image(type="pil", label="Foto de la hoja (opcional)",
                                   height=260, sources=["upload", "clipboard"])
                    txt = gr.Textbox(label="Tu pregunta", lines=3,
                                     placeholder="Ej.: Esta hoja es de mi tomate bajo "
                                                 "invernadero. ¿Qué manejo me recomiendan?")
                    btn = gr.Button("Consultar", variant="primary")
                with gr.Column(scale=2):
                    out = gr.Markdown(label="Respuesta")
                    with gr.Accordion("Cómo llegó a esta respuesta", open=False):
                        resumen = gr.Markdown()
                        crudo = gr.Code(label="detalle (JSON)", language="json")
            ej = _ejemplos()
            if ej:
                gr.Examples(examples=ej, inputs=[img, txt], label="Casos del eval set")
            btn.click(consultar, [img, txt], [out, resumen, crudo],
                      api_name="consultar", concurrency_limit=1)
            txt.submit(consultar, [img, txt], [out, resumen, crudo], concurrency_limit=1)

        with gr.Tab("Estado"):
            gr.Markdown("Health check del servicio. El mismo objeto se expone como API en "
                        "`/gradio_api/call/health`.")
            with gr.Row():
                refrescar = gr.Button("Refrescar", variant="secondary")
            salud_json = gr.JSON(value=estado, label="salud")
            gr.Markdown("#### Últimas peticiones (una línea JSON por petición)")
            logs = gr.Code(value=ultimas_lineas, language="json", label="logs/peticiones.jsonl")
            refrescar.click(estado, None, salud_json, api_name="health")
            refrescar.click(ultimas_lineas, None, logs)

        with gr.Tab("Cómo funciona"):
            gr.Markdown(como)

    return demo


if __name__ == "__main__":
    calentar()
    construir().queue(max_size=16).launch(
        theme=gr.themes.Soft(),
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=os.environ.get("AGRO_SHARE", "0") == "1"
    )