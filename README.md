# 🌿 Recomendador de manejo agronómico

### **"Esta hoja de mi cultivo se ve así — ¿qué tiene y qué le hago?"**

[![Abrir en Hugging Face](https://img.shields.io/badge/🤗%20Space-Manco312%2Fagroai-yellow)](https://huggingface.co/spaces/Manco312/agroai)

**→ Sistema en producción: <https://huggingface.co/spaces/Manco312/agroai>**

**Tópicos Especiales y Aplicaciones en IA · Universidad EAFIT · M1 → M5**
**Equipo:** Luciana Hoyos · Sara López · Juan Carlos Citelly · Santiago Manco Maya

---

Un agricultor no sabe que su papa tiene *Phytophthora infestans*. Sabe que las hojas se le
están poniendo negras y tiene el teléfono en la mano. Este sistema le pide una foto y le
devuelve el manejo que dice la ficha técnica, con su fuente — o le dice que no sabe.

> **Nunca entrega dosis ni periodos de carencia.** Ninguna de las 50 fichas del corpus
> contiene una cifra de dosis, así que toda dosis en una respuesta estaría inventada. El
> sistema las bloquea y redirige a la etiqueta del producto registrado ante el ICA.
> **Recomienda manejo agronómico; no sustituye a un agrónomo ni al ICA.**

## Qué hace, en una frase

Una foto pasa por **CLIP ViT-B/32 con una cabeza lineal de 38 clases** (91,4 % de top-1 sobre
760 imágenes retenidas); una **válvula de cuatro guardas** decide si esa etiqueta es creíble;
la etiqueta elige la **ficha técnica** entre 151 pasajes de 50 documentos; un modelo de
lenguaje **redacta sobre esos pasajes**; y un **filtro de salida determinista** retira
cualquier receta sobre hoja sana, cualquier receta sin diagnóstico y cualquier dosis.

## El resultado que define el proyecto

M4 terminó con un fallo concreto: sobre una hoja de tomate **sana**, el sistema escribió
*"se recomienda fungicida foliar… generalmente se usa entre **50-100 gramos por hectárea**"*
justo debajo del encabezado que decía que no había nada que tratar. Y la regla de dominio lo
contaba como acierto.

M5 reprodujo ese caso y lo cerró. Sobre 102 respuestas medidas:

| | sin filtro de salida | con filtro |
|---|---:|---:|
| respuestas con dosis numérica | 1 | **0** |
| respuestas que recetan sobre hoja sana | 1 | **0** |
| falsos positivos sobre 47 respuestas de referencia | — | **0** |

## Qué hay en este repo

| Carpeta | Qué es |
|---|---|
| **[`M5/`](M5/)** | **La entrega final.** El sistema completo como paquete desplegable, la app, las mediciones y el reporte. |
| [`M5/REPORTE_FINAL.md`](M5/REPORTE_FINAL.md) | **El reporte del proyecto M1→M5**: problema, arquitectura, decisiones, scorecard contra las líneas base, técnica de eficiencia con su contrato, ablaciones, errores, límites y ética. |
| [`M5/README.md`](M5/README.md) | Cómo correrlo, cómo reproducir cada número y cómo se despliega. |
| [`M5/resultados/`](M5/resultados/) | Los `.json`, `.csv` y `.md` de cada corrida, y el plan B de la demo. |
| [`modulos_anteriores/`](modulos_anteriores/) | Los notebooks de M1, M2, M3 y M4 **sin modificar**, con sus datos y scorecards. |

## El recorrido

Cada módulo movió el cuello de botella en vez de resolverlo del todo. Esa es la historia.

| | Qué construyó | Dónde quedó el problema |
|---|---|---|
| **M1** | fine-tuning LoRA de Qwen2.5-0.5B sobre 38 fichas | repite la forma pero no sabe de lo que habla |
| **M2** | el arnés de evaluación: 3 dimensiones, juez por valor esperado, regla de dominio por caso | **conocimiento**: no tiene las fichas delante |
| **M3** | RAG agéntico: 50 PDF, BM25 + denso (RRF), reranker, 3 herramientas, ReAct | **generación**: llega la ficha correcta y el 0.5B no la convierte en acierto |
| **M4** | el componente visual: CLIP + cabeza lineal + válvula de 4 guardas | **generación, confirmada**: con el diagnóstico correcto en el prompt, igual recetó sobre una hoja sana |
| **M5** | despliegue, eficiencia medida, filtro de salida y observabilidad | se ataca de frente: guardas deterministas + un redactor que no tiene ese problema |

## Arranque rápido

```bash
cd M5
pip install -r requirements.txt gradio==6.31.0

python scripts/test_filtro.py    # pruebas del filtro de salida (1 s, sin modelos)
python scripts/humo.py           # prueba de humo de las cuatro rutas
python app.py                    # la app en http://localhost:7860
```

Sin `GROQ_API_KEY` el sistema **no se cae**: usa el modelo propio afinado en M1. Los detalles
de configuración, despliegue y reproducción están en [`M5/README.md`](M5/README.md).

## Lo que no sabemos

**PlantVillage no es el campo.** El 91,4 % se midió sobre hojas arrancadas, con fondo
uniforme y luz de laboratorio. Ante una foto de campo real la válvula puede dispararse y
dejar al agricultor sin diagnóstico justo en el caso de uso que importa. No está medido, y
es el límite más serio del proyecto. El [reporte final](M5/REPORTE_FINAL.md#10--límites) lista
los demás.
