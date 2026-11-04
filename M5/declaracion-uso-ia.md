# Declaración de uso de Inteligencia Artificial — Entrega M5

**Equipo:** Luciana Hoyos · Sara López · Juan Carlos Citelly · Santiago Manco Maya

---

## Herramienta utilizada

Se utilizó **Claude (Anthropic)** como asistente durante el desarrollo de esta entrega, igual
que en M1–M4. A continuación se detalla en qué tareas se usó, en cuáles no, y qué quedó bajo
revisión y decisión directa del equipo.

## Tareas en las que se usó IA

- **Extracción del sistema fuera de los notebooks.** M4 cargaba el sistema ejecutando las
  celdas de `notebookM3.ipynb` en un espacio de nombres propio; eso no se puede desplegar.
  Claude ayudó a portar las funciones de M1–M4 al paquete `agro/` (catálogo, troceo,
  recuperación híbrida, reranking, las tres herramientas, el ReAct, `ver()` y la válvula)
  **conservando la lógica**, y a escribir las verificaciones que prueban que se conservó:
  los 151 chunks y las seis consultas de retrieval con salida registrada en `notebookM3`.
- **El código de infraestructura de M5:** los cuatro *backends* del redactor (fp16 en GPU,
  fp32 en CPU, int8 dinámico y NF4 con bitsandbytes), la app de Gradio, el health check, el
  logging JSON por petición, la integración con W&B y el script que arma la carpeta del Space.
- **El arnés de medición:** el banco de cuantización en dos pasadas (capturar los prompts una
  vez y después generar con cada variante sobre los mismos prompts), las dos ablaciones y el
  script de puntuación.
- **Documentación y redacción:** comentarios del código, `README.md` y `REPORTE_FINAL.md`.
  Claude redactó los textos a partir de los números de las corridas del equipo.
- **Depuración.** En esta entrega, sobre todo de memoria: el proceso moría sin excepción al
  cargar el juez de 1.5B junto a CLIP y el cross-encoder, y la primera versión del banco
  volvía a llamar a `responder()` por caso para recomponer la respuesta, manteniendo cargados
  ~1.5 GB de modelos que no usaba y midiendo la latencia con la máquina en *swap*. Claude
  diagnosticó las dos cosas y propuso separar la puntuación en un proceso aparte y grabar el
  prefijo determinista en la primera pasada.

## Tareas que NO se delegaron a la IA

- **Las decisiones de arquitectura del despliegue.** Que el sistema desplegado siga siendo el
  **modelo propio del curso** (Qwen2.5-0.5B + el adaptador LoRA de M1) cuantizado a int8, y no
  una API externa, aunque el despliegue sugerido ofrecía Groq. Groq se usó en M3 como tercer
  juez y se deja disponible como *backend* opcional, pero no es el sistema que se entrega:
  cambiarlo por un modelo de 120B haría que la URL pública no demostrara nada de M1–M4.
- **El contrato de la cuantización.** Qué se promete y qué se acepta perder (`C1`–`C6` en
  `scripts/bench_contrato.py`) se escribió y se versionó **antes** de correr la primera
  medición, con la regla explícita de que una variante que no lo cumpla no se despliega.
- **El diseño del filtro de salida.** Las tres reglas y, sobre todo, la justificación de la
  tercera: la comprobación de que **ninguna de las 50 fichas del corpus contiene una dosis
  numérica**, que es lo que convierte "bloquear las dosis" en un hecho verificable y no en una
  opinión agronómica. El script de construcción del índice falla si eso deja de ser cierto.
- **La ejecución real de las mediciones.** Todas las corridas (cuantización en CPU y en GPU,
  las dos ablaciones y las pruebas de humo) se hicieron en el entorno del equipo. Ningún
  número de este reporte fue generado ni estimado por la IA.
- **La lectura honesta de los resultados y qué reportar.** En particular, reportar que la ruta
  determinista rompía la abstención en los cultivos no cubiertos antes de corregirla, y
  reportar el coste real de la cuantización en vez de presentarla sólo como un ahorro.
- **El despliegue en sí.** La creación del Space, la carga de los secretos y la verificación
  de la URL pública desde otro dispositivo las hizo el equipo.

## Verificación

Todo lo que esta entrega afirma está respaldado por un artefacto en el repo: los `.json` y
`.csv` de `resultados/`, el `manifiesto.json` del índice con los hashes de cada artefacto y
las versiones del entorno, y el log JSONL de las peticiones. El `README.md` dice qué comando
reproduce cada número.
