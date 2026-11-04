# Declaración de uso de Inteligencia Artificial — Entrega M3

**Equipo:** Luciana Hoyos · Sara López · Juan Carlos Citelly · Santiago Manco Maya

---

## Herramienta utilizada

Se utilizó **Claude (Anthropic)** como asistente durante el desarrollo de esta entrega.
A continuación se detalla en qué tareas se usó, en cuáles no, y qué quedó bajo revisión y
decisión directa del equipo.

## Tareas en las que se usó IA

- **Código técnico y repetitivo del pipeline RAG:** ingesta de los PDF con `pypdf`, chunking
  por sección, índice denso en Chroma, BM25, fusión RRF, reranking con cross-encoder, el bucle
  ReAct del agente, el RAGAS casero y la evaluación de retrieval (hit@3 / precision@3).
- **Integración del tercer juez vía API (Groq):** llamadas a `openai/gpt-oss-120b`, manejo de
  errores y de límites de tasa.
- **Generación del corpus en PDF:** el script que convierte las fichas a PDF con `reportlab`.
- **Redacción de contenido base:** Claude redactó el texto de las 12 fichas técnicas de
  Colombia (`datos/corpus_ica_colombia.json`), a partir de información pública de ICA,
  AGROSAVIA, Cenicafé y Fedecacao, y los 14 casos nuevos del eval set
  (`eval_set_m3_extra.json`). Las fichas no son documentos oficiales y así lo declara cada PDF.
- **Documentación del código y redacción general:** comentarios del código, textos markdown
  del notebook y README. Claude redactó de manera profesional los textos, incluidas la lectura
  honesta y las limitaciones, a partir de los resultados de la corrida del equipo.
- **Depuración:** después de la primera corrida (v1), detección y corrección de errores. Entre
  ellos: el modelo de Groq retirado que dejaba vacío el tercer juez, el detector de abstención
  que no reconocía la válvula de escape ni los plurales en prescripción, el planificador que
  tomaba "envés" como cultivo y las respuestas que inventaban pares "Pregunta: … Respuesta:".

## Tareas que NO se delegaron a la IA

- **Estructuración base del notebook:** El equipo estructura el notebook tanto con sus secciones
de código como con sus secciones de markdown.
- **Definición del alcance y de las decisiones de diseño:** que el corpus fueran documentos
  PDF, que se reutilizaran los datos de M1 y M2, que la herramienta del agente aportara valor
  real al dominio (se descartó una calculadora genérica) y que los notebooks de M1 y M2 no se
  modificaran.
- **La ejecución real del notebook:** correrlo de principio a fin, descargar los modelos y
  llamar a la API de Groq con la clave del equipo se hizo en el entorno del equipo. Nada fue
  generado ni simulado por la IA. Los números del scorecard, de RAGAS y del tercer juez que
  están en el repo provienen de esas corridas (v1 y v2).
- **La revisión de resultados y la decisión de iterar:** el equipo revisó los resultados de la
  v1, decidió aplicar los arreglos de la v2 y volvió a correr el notebook para medirlos.
- **La elección de los modelos:** el planificador, los jueces locales y el juez de API se
  evaluaron y aceptaron con base en las restricciones de la GPU del equipo (6 GB) y del plan
  gratuito de Groq.
