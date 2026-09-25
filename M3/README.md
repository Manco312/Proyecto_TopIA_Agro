# RAG agéntico y evaluación con RAGAS · Entrega M3 (15%)

**Recomendador de manejo agronómico para enfermedades en hojas de plantas, ahora con una biblioteca consultable**

**Tópicos Especiales y Aplicaciones en IA · Universidad EAFIT · Módulo 3 — RAG**

**Equipo:** Luciana Hoyos · Sara López · Juan Carlos Citelly · Santiago Manco Maya

---

## En una frase

> **Le damos al recomendador de M1 (Qwen2.5-0.5B + LoRA) una biblioteca de 50 PDF que puede
> consultar antes de responder: los 38 documentos de su dominio de entrenamiento y 12 fichas de
> manejo para Colombia (ICA, AGROSAVIA, Cenicafé, Fedecacao) que el modelo nunca vio.
> Construimos el pipeline RAG completo, le damos herramientas de dominio en un bucle ReAct y lo
> medimos con el harness de M2 (corregido), RAGAS y un tercer juez vía API.**

**Resultado principal:** el **RAG agéntico** sube los aciertos de dominio gold de **2/22 a
9/22**, el patógeno correcto de **1/20 a 15/20** y deja en **0/20** los patógenos de otro
cultivo. El RAG ingenuo y el avanzado recuperan el documento correcto (hit@3 = 100 % con
reranking), pero el modelo de 0.5B no convierte ese contexto en aciertos. El cuello de botella
pasó del *conocimiento* a la *generación*.

## Qué hay en esta entrega

| Archivo | Qué es |
|---|---|
| [`notebookM3.ipynb`](notebookM3.ipynb) | **La entrega ejecutable.** Pipeline RAG, herramientas y ReAct, harness sobre 4 sistemas, RAGAS, tercer juez, lectura honesta (§21) y limitaciones (§23). Ya viene ejecutado (corrida v2). |
| [`eval_set.json`](eval_set.json) | Los 13 casos de M2, **sin tocar**. |
| [`eval_set_m3_extra.json`](eval_set_m3_extra.json) | **14 casos nuevos:** 12 gold de fuente externa (`gold-ica-*`) y 2 adversariales (`adv-04` ambigüedad sin cultivo, `adv-05` premisa falsa). |
| `eval_set_m3.json` | Los 27 casos combinados que usó la corrida (lo genera el notebook). |
| [`datos/pdfs/`](datos/pdfs/) | **El corpus RAG:** `entrenamiento/` (38 PDF, uno por clase de PlantVillage) y `externo/` (12 fichas de Colombia). |
| [`datos/corpus_ica_colombia.json`](datos/corpus_ica_colombia.json) | Metadatos y texto fuente de las 12 fichas externas (con su `etiqueta`). |
| `scorecard_rag_m3.csv` / `.json` | **El scorecard de M3:** 4 sistemas × 27 casos. El `.json` trae config, versiones, `revision` de cada modelo y el detalle caso por caso. |
| `comparacion_v1_v2_m3.csv` | v1, v1 re-puntuada y v2 por sistema: separa el efecto del medidor del efecto del sistema (§16b). |
| `retrieval_hitk_m3.csv` | hit@3 y precision@3 por método y configuración de chunking, sin LLM (§10b). |
| `ragas_m3.csv` | RAGAS casero por caso: faithfulness, context precision y recall, answer relevancy (§18). |
| `anclaje_juez_m3.csv` / `juez_api_m3.csv` | Tercer juez (Groq, `openai/gpt-oss-120b`): sondas sintéticas y respuestas reales (§19). |
| `RUBRICA.md` / `RUBRICA_snapshot_m3.txt` | Rúbricas del juez (heredadas de M2) y la copia usada en la corrida. |
| [`resultados_v1/`](resultados_v1/) | La primera corrida completa (v1), guardada para la comparación de §16b. |
| `mi-modelo-lora/` | El adaptador LoRA de M1: el **sistema** sobre el que se construye el RAG. |
| `notebookM1.ipynb` / `notebookM2.ipynb` | Las entregas anteriores, para que M3 sea autocontenido (sin modificar). |
| `wandb/` | Registro offline de la corrida en Weights & Biases (opcional, §20). |
| [`declaracion-uso-ia.md`](declaracion-uso-ia.md) | Declaración de uso de IA en esta entrega. |

## Qué corrige esta entrega respecto al feedback de M2

| Feedback de M2 | Qué hicimos |
|---|---|
| Los gold salían de la misma base con la que se entrenó el modelo | **12 gold nuevos** cuya respuesta vive en fichas de Colombia que el modelo nunca vio. Dos (café y cacao) están **fuera de las 38 clases**. |
| 13 casos son pocos; 20–25 gold darían sub-criterios estables | **22 gold + 5 adversariales = 27 casos.** |
| El juez de 1.5B ancla en 3; probar un juez grande vía API | **Tercer juez** `openai/gpt-oss-120b` vía Groq, de otra familia que Qwen, sobre sondas sintéticas y respuestas reales (§19). |
| `menciona_patogeno` acepta el nombre del cultivo dentro del binomio | Se excluyen los tokens del cultivo, con prueba de regresión sobre `gold-03` (§5). Sigue siendo incompleto: ver Limitaciones. |

## El sistema

```
pregunta ──► [BM25 + denso (Chroma)] ──RRF──► [cross-encoder rerank] ──► top-3 chunks ──► Qwen2.5-0.5B + LoRA (M1) ──► respuesta
                  técnica avanzada 1               técnica avanzada 2                        prompt con válvula de escape

agéntico:  planificador (Qwen2.5-1.5B) ─ReAct─► diagnostico_diferencial ─► consultar_ficha ─► modelo de M1 redacta
                                                 (¿cultivo cubierto?,       (ficha exacta
                                                  candidatos del cultivo)    por etiqueta)
```

- **Corpus:** 50 PDF ingeridos con `pypdf` y cortados por sección de la ficha, con una cabecera
  de contexto (título, cultivo y agente causal).
- **Técnica avanzada 1, hybrid search:** BM25 con tokens normalizados más embeddings densos
  (`paraphrase-multilingual-MiniLM-L12-v2` en Chroma), fusionados con Reciprocal Rank Fusion.
- **Técnica avanzada 2, reranking:** cross-encoder multilingüe
  `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` sobre 10 candidatos, del que salen los 3 finales.
- **Tool use + ReAct** (herramientas elegidas porque atacan fallas que M2 midió, no por
  cumplir):
  - `diagnostico_diferencial`: dice si el cultivo está cubierto y qué enfermedades *de ese
    cultivo* calzan con los síntomas. Le da al sistema el modo "no sé / ¿qué cultivo?" que en
    M2 fue 0/3.
  - `consultar_ficha`: trae la ficha exacta por etiqueta de PlantVillage. Es la interfaz
    prevista con el clasificador de imágenes de M4.
  - `buscar_en_fichas`: búsqueda libre en el índice híbrido.
- **La respuesta final siempre la redacta el modelo de M1.** El planificador solo decide qué
  herramientas llamar.

## Resultados (corrida v2)

### Scorecard de dominio (Dimensión 3) y costo

| Sistema | Aciertos gold | … externos | Patógeno correcto | Patógeno de otro caso | Formato | Cobertura | Adversariales | s / consulta |
|---|---|---|---|---|---|---|---|---|
| 0 · baseline (M1 sin RAG) | 2/22 | 1/12 | 1/20 | 4/20 | 21/22 | 0.15 | 0/5 | 12.5 |
| 1 · RAG ingenuo (denso) | 4/22 | 2/12 | 6/20 | 3/20 | 16/22 | 0.24 | 2/5 | 13.1 |
| 2 · RAG avanzado (híbrido + rerank) | 3/22 | 2/12 | 12/20 | 3/20 | 18/22 | 0.30 | 1/5 | 15.2 |
| 3 · RAG agéntico (herramientas + ReAct) | **9/22** | **4/12** | **15/20** | **0/20** | 17/22 | **0.54** | 3/5\* | 36.7 |

\* Los aciertos en `adv-01` de los tres RAG son un **artefacto del detector**: acepta "Cenicafé"
como señal de abstención y el RAG lo cita como fuente. El conteo real es ingenuo 1/5, avanzado
0/5 y agéntico **2/5** (paraquat y "¿qué cultivo?").

Dimensiones 1 y 2 (gold): similitud por embeddings 0.665 / 0.667 / 0.703 / **0.761**; ROUGE-L
0.225 / 0.194 / 0.212 / **0.282**; juez local 2.94 / 2.73 / 2.87 / 2.98. El juez local **no
distingue** entre sistemas (ver abajo).

### Retrieval, sin LLM (§10b)

| Método (chunking por sección) | hit@3 M2 | hit@3 externos | precision@3 M2 | precision@3 externos |
|---|---|---|---|---|
| Denso (ingenuo) | 3/10 | 6/12 | 0.10 | 0.39 |
| BM25 | 10/10 | 11/12 | 0.57 | 0.39 |
| Híbrido (RRF) | 10/10 | 10/12 | 0.40 | 0.42 |
| **Híbrido + rerank** | **10/10** | **12/12** | **0.47** | **0.56** |

El diagnóstico diferencial acierta la enfermedad en top-1 en **19/22** casos con cross-encoder,
frente a 13/22 con embeddings.

### RAGAS casero (§18)

| Sistema | faithfulness | context_precision | context_recall | answer_relevancy |
|---|---|---|---|---|
| RAG ingenuo | 0.535 | 0.287 | 0.182 | 0.624 |
| RAG avanzado | 0.520 | **0.530** | 0.182 | 0.502 |

El reranking casi duplica la precisión del contexto, pero la fidelidad de la respuesta no sube.

### Tercer juez vía API (§19)

| Juez · gold | Baseline | RAG avanzado |
|---|---|---|
| Local `Qwen2.5-1.5B` | 2.94 | 2.87 |
| API `gpt-oss-120b` | 1.32 | **2.18** |

- **No hay anclaje en 3.** En las sondas sintéticas, el juez de API da 5 / 1 / 1 (buena / vaga /
  disparate). Pero tampoco distingue "vaga" de "disparate".
- **Solo el juez de API ve la mejora del RAG.** Coincide con la Dimensión 3: pone 2.5 de
  promedio a los aciertos y 1.73 a los fallos. Su acuerdo con el juez local es nulo (Spearman
  0.007).

### v1 → v2 (§16b)

La primera corrida (v1) mostró que el RAG traía conocimiento pero no aciertos. Los arreglos de
v2 (chunking por sección, BM25 normalizado, formato de M1 en la instrucción, corte en
"Pregunta:", cross-encoder en el diagnóstico) llevaron el agéntico de 1/22 a 9/22 (con el
medidor ya corregido). Además, recuperaron el formato (9 → 17 y 9 → 18/22) y eliminaron las
respuestas desbocadas (4 / 5 / 3 → 0).

## Lectura honesta (resumen; completa en §21 del notebook)

- **Qué técnica movió qué:**
  - **Hybrid + reranking** mueve el *retrieval* (precision@3 de 0.10 a 0.47–0.56; patógeno
    correcto de 6 a 12/20), pero **no los aciertos** (4 → 3/22).
  - **Las herramientas** sí mueven los aciertos (9/22), porque le entregan al modelo una sola
    ficha, la correcta, filtrada por cultivo.
  - **El chunking por sección** fue un intercambio: mejora en las fichas externas y empeora en
    los documentos de M2.
- **Dónde falla todavía:**
  - 4 de los 13 fallos del agente son **solo de formato** (el modelo de 0.5B rompe la
    plantilla o deja escapar texto del prompt).
  - Las **dos premisas falsas fallan en los cuatro sistemas**.
  - El agente cuesta **~3× la latencia** del baseline (36.7 s por consulta).
- **Sobre la medición:**
  - El juez local de 1.5B no sirve para comparar sistemas en M3. La evidencia es la Dimensión 3
    más el juez de API.
  - La abstención del agente la produce un guardrail determinista, no el modelo.

## Limitaciones principales (completas en §23)

- Las fichas externas **no son documentos oficiales**: se redactaron para el ejercicio, con
  asistencia de IA, y lo declaran en cada PDF.
- La **v2 se ajustó mirando la v1 sobre los mismos 27 casos**, y no hay un conjunto apartado
  para validar.
- Con n = 22 gold, **un caso vale 4.5 puntos porcentuales**, y cada sistema se corrió una sola
  vez.
- Detectores regex con aciertos falsos conocidos (`adv-01`) y un fix de `menciona_patogeno`
  incompleto (`mosaic` ⊂ "mosaico").
- El juez de API solo puntuó al baseline y al avanzado. RAGAS solo cubre al ingenuo y al
  avanzado.
- La latencia se midió en una laptop con 6 GB de VRAM.

## Cómo se corre (reproducibilidad)

1. Abrir [`notebookM3.ipynb`](notebookM3.ipynb) en Colab (T4 recomendada) o en Jupyter local con
   GPU.
2. *(Opcional, para la §19)* Crear `M3/.env` con `GROQ_API_KEY=...` (ya está en `.gitignore`) o
   configurar la clave en los *Secretos* de Colab. Sin clave, la §19 se omite sola.
3. **Runtime → Run all.** Los modelos se descargan del Hub de Hugging Face. Si faltan los PDF, el
   notebook los regenera con `reportlab`.
4. Produce en `M3/` los archivos de resultados de la tabla de arriba. La v1 queda intacta en
   `resultados_v1/`.

**Determinismo:** `SEED = 42`, decodificación greedy en el sistema y el planificador, jueces
locales por valor esperado sobre logits, y versiones de librerías y `revision` de cada modelo
guardadas en `scorecard_rag_m3.json`. Lo único no determinista es el juez de API (servicio
externo), que por eso se reporta aparte.

## Referencias

- Modelo base / sistema: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct (+ adaptador LoRA de M1)
- Planificador y juez principal: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- Juez de control: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct
- Tercer juez: `openai/gpt-oss-120b` vía https://console.groq.com
- Embeddings: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Reranker: https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
- Dataset (taxonomía de clases): https://github.com/spmohanty/plantvillage-dataset
- Fuentes agronómicas por clase: [`datos/REFERENCIAS.md`](datos/REFERENCIAS.md)
- Labs del curso usados como referencia: [`ejemplosM3/`](ejemplosM3/) (S07, S08, S10)
