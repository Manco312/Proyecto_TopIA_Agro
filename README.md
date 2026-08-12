# Clasificador y recomendador para enfermedades en hojas de plantas

**Entrega M1 · Tópicos Especiales y Aplicaciones en IA · Universidad EAFIT**

**Equipo:** Luciana Hoyos · Sara López · Juan Carlos Citelly · Santiago Manco Maya

---

## 1. Descripción del proyecto

La visión completa del proyecto es un sistema de dos etapas para apoyar a
agricultores en el manejo de enfermedades foliares:

1. **Clasificador (computer vision, fase posterior del curso):** a partir de una
   foto de una hoja, predice a cuál de las 38 clases de PlantVillage pertenece
   (26 combinaciones cultivo-enfermedad/plaga + 12 estados sanos).
2. **Recomendador (NLP, esta entrega):** a partir de esa etiqueta (o de una
   pregunta en lenguaje natural sobre ella), genera una recomendación de manejo
   agronómico en español — identificación, acción inmediata y prevención.

**Esta entrega M1 cubre únicamente el recomendador de texto.** Por instrucción
del curso, en esta fase no se usa ninguna imagen del dataset PlantVillage, solo
sus **etiquetas**. El recomendador es un modelo de lenguaje pequeño (Qwen2.5-0.5B-Instruct)
adaptado con **fine-tuning LoRA** sobre un dataset de instrucciones construido
específicamente para esta tarea.

## 2. Estructura del repositorio

```
proyecto_M1/
├── README.md                                    <- este archivo
├── notebook.ipynb                                <- notebook de la entrega (ejecutado)
└── datos/
    ├── dataset_finetuning_plantvillage.jsonl     <- 204 ejemplos instrucción → respuesta
    ├── base_conocimiento_plantvillage.json       <- base de conocimiento cruda, con fuente por clase
    ├── base_conocimiento_plantvillage.csv        <- la misma base, en formato tabla
    └── REFERENCIAS.md                            <- todas las fuentes citadas
```

## 3. Cómo correr el notebook

1. Abre `notebook.ipynb` en Google Colab (recomendado) o localmente con Jupyter.
2. Activa GPU: `Runtime → Change runtime type → T4 GPU` (en Colab). Sin GPU el
   fine-tuning es muy lento.
3. Asegúrate de que la carpeta `datos/` esté en la misma ubicación que el notebook
   (si subes solo el `.ipynb` a Colab, sube también la carpeta `datos/` a la sesión,
   o móntala desde Google Drive).
4. Corre las celdas en orden. La primera celda instala todas las dependencias
   (`transformers`, `datasets`, `peft`, `accelerate`, `evaluate`, `bitsandbytes`).
5. El notebook importa el modelo base directamente desde el Hub de Hugging Face
   (`Qwen/Qwen2.5-0.5B-Instruct`) — no requiere ningún archivo de pesos local.

## 4. Selección y justificación del modelo base

Se compararon 4 modelos pequeños (0.5B–3.8B parámetros), evaluando tamaño,
licencia, soporte de español y comportamiento del tokenizador sobre texto
agronómico en español (evidencia empírica, no solo lectura de model card).

| Modelo | Parámetros | Licencia | Español |
|---|---|---|---|
| **Qwen2.5-0.5B-Instruct (elegido)** | 0.5B | Apache 2.0 | Sí, explícito en el entrenamiento |
| Qwen2.5-1.5B-Instruct (respaldo) | 1.5B | Apache 2.0 | Sí, misma familia |
| Phi-3-mini-4k-instruct | 3.8B | MIT | Secundario, mayormente inglés |
| TinyLlama-1.1B-Chat | 1.1B | Apache 2.0 | Pobre, casi exclusivamente inglés |

**Se eligió `Qwen/Qwen2.5-0.5B-Instruct`, importado desde el Hub de Hugging Face**,
por ser el más liviano de la comparación (entrena rápido con LoRA en una GPU
gratuita de Colab), tener licencia permisiva, soporte explícito de español, y el
menor ratio de tokens-por-palabra sobre frases agronómicas en español frente a
los demás candidatos (ver sección 1 del notebook para la medición completa). El
detalle de la comparación, incluyendo el código que mide el tokenizador sobre el
dominio, está en la sección 1 del `notebook.ipynb`.

## 5. Dataset: construcción y documentación

### Origen de las etiquetas
Las 38 clases vienen del dataset público **PlantVillage** (Hughes & Salathé,
2015; Mohanty et al., 2016), 54,306 imágenes de hojas sanas y enfermas en 14
especies de cultivo. En esta entrega **no se usa ninguna imagen**, solo la
taxonomía de labels (`Cultivo___Enfermedad`).

### Construcción del dataset de fine-tuning
No existe un dataset público de recomendaciones agronómicas en español para las
etiquetas de PlantVillage. El dataset (`datos/dataset_finetuning_plantvillage.jsonl`)
es **derivado y curado por el equipo**:

1. Se construyó una base de conocimiento (`datos/base_conocimiento_plantvillage.json`)
   con, para cada una de las 38 clases: cultivo, patógeno/causa, identificación,
   tratamiento, prevención, y **la fuente exacta** de donde salió esa información
   (mayoritariamente servicios de extensión agrícola universitaria: UMN Extension,
   Penn State, MSU, UF/IFAS, Cornell, UC IPM, entre otros — lista completa en
   `datos/REFERENCIAS.md`).
2. Sobre esa base, se generaron entre 4 y 6 variantes de pregunta por clase
   (plantillas de instrucción distintas), para un total de **204 ejemplos**
   `instrucción → respuesta`.
3. Split reproducible train/val/test (70/15/15 aprox.) agrupado por clase con
   seed fija, garantizando que las 38 clases aparezcan en los tres splits.

### Por qué es representativo de la tarea
- Cubre las 38 clases completas, no un subconjunto — el recomendador debe poder
  responder ante cualquier etiqueta que a futuro prediga el clasificador de imágenes.
- Múltiples formulaciones de pregunta por clase enseñan al modelo el *formato*
  de respuesta esperado, no una redacción memorizada.
- Las respuestas siguen una estructura consistente y accionable: identificación,
  acción inmediata, prevención — justo lo que un agricultor necesita.

### Limitaciones documentadas
- Recomendaciones generales/educativas; no sustituyen a un agrónomo en campo ni
  dan dosis exactas de agroquímico.
- Fuentes primarias mayoritariamente de EE. UU.; para Colombia debería
  contrastarse con guías del ICA o gremios locales.
- ~5-6 ejemplos por clase en promedio — suficiente para que el modelo aprenda el
  formato con LoRA, pero limitado en variedad real de contenido por clase.

Ver `datos/REFERENCIAS.md` para el listado completo de fuentes citadas.

## 6. Fine-tuning

- **Método:** LoRA (Low-Rank Adaptation) vía la librería `peft`, sobre
  `Qwen/Qwen2.5-0.5B-Instruct`.
- **Hiperparámetros:** `r=16`, `lora_alpha=32`, `target_modules=[q_proj, k_proj,
  v_proj, o_proj]`, `lora_dropout=0.05`, 6 épocas, `learning_rate=2e-4`, seed fija
  (42) para reproducibilidad. Justificación de `r=16` (más alto que el default de
  clase, 8): el dataset cubre 38 clases distintas en vez de un solo dominio, y un
  rango mayor ayuda al modelo a diferenciar mejor entre etiquetas.
- **Entrenamiento y evaluación:** `Trainer` de `transformers`, con `eval_strategy="epoch"`
  sobre el split de validación.

## 7. Baseline vs. resultado

El notebook mide:
1. Una comparación **puntual**: la misma pregunta del dominio, respondida por el
   modelo antes y después del fine-tuning (Lab A → Lab C).
2. Una comparación **cuantitativa**: ROUGE-1/2/L del modelo antes y después,
   calculado sobre el mismo conjunto de prueba (`test_df`), para sustentar con
   números si el fine-tuning mejoró la tarea, no solo con una lectura cualitativa
   de un único ejemplo.

Los valores concretos (números de ROUGE y las respuestas antes/después) quedan
documentados en la celda de cierre del `notebook.ipynb`, sección "Para la entrega
M1 — documentación del equipo", una vez ejecutado.

## 8. Trabajo futuro (fuera del alcance de M1)

- Integrar el clasificador de imágenes (computer vision) que prediga la etiqueta
  de PlantVillage a partir de una foto real de la hoja, y conectarlo con este
  recomendador de texto.
- Extender/contrastar la base de conocimiento con fuentes específicas de Colombia
  (ICA, gremios agrícolas locales).
- Evaluación más allá de ROUGE (evaluación humana o LLM-as-judge), dado que ROUGE
  no captura corrección agronómica del contenido, solo solape léxico.

## 9. Referencias principales

- Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on
  plant health...* arXiv:1511.08060.
- Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using Deep Learning for
  Image-Based Plant Disease Detection*. Frontiers in Plant Science, 7:1419.
  https://doi.org/10.3389/fpls.2016.01419
- Dataset de imágenes en el Hub: https://huggingface.co/datasets/mohanty/PlantVillage
- Modelo base: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- Listado completo de fuentes de manejo agronómico por enfermedad: ver `datos/REFERENCIAS.md`.
