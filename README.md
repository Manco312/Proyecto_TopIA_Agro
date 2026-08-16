# Clasificador y recomendador para enfermedades en hojas de plantas

**Entrega M1 · Tópicos Especiales y Aplicaciones en IA · Universidad EAFIT**

**Equipo:** Luciana Hoyos · Sara López · Juan Carlos Citelly · Santiago Manco Maya

---

## Descripción del proyecto

La visión completa del proyecto es un sistema de dos etapas: un **clasificador**
de imágenes (computer vision, fase posterior del curso) que prediga a cuál de
las 38 clases de PlantVillage pertenece una hoja, y un **recomendador** (NLP,
esta entrega) que a partir de esa etiqueta genere una recomendación de manejo
agronómico en español.

<img width="964" height="824" alt="image" src="https://github.com/user-attachments/assets/342486f5-da89-4abd-b9f9-adfb1362d38d" />

Esta entrega M1 cubre solo el recomendador de texto: fine-tuning con LoRA de
`Qwen2.5-0.5B-Instruct` sobre un dataset de instrucciones construido por el
equipo. No se usa ninguna imagen para entrenar, solo las etiquetas de
PlantVillage.

Todo el detalle técnico (justificación del modelo, construcción del dataset,
hiperparámetros, baseline vs. resultado, lectura honesta de las métricas) está
documentado paso a paso en `notebook.ipynb` — este README solo da el contexto
general y cómo correrlo.

## Estructura del repositorio

```
proyecto_M1/
├── README.md
├── declaracion-uso-ia.md
├── notebook.ipynb
└── datos/
├── color/ <- imágenes de PlantVillage, descargar antes de ejecutar
│ (https://huggingface.co/datasets/Jackieeeeee/plantvillage-raw-color-7f7ecc7)
├── dataset_finetuning_plantvillage.jsonl
├── base_conocimiento_plantvillage.json
├── base_conocimiento_plantvillage.csv
└── REFERENCIAS.md
```

## Cómo correr el notebook

1. Abre `notebook.ipynb` en Google Colab (recomendado) o Jupyter local.
2. Activa GPU: `Runtime → Change runtime type → T4 GPU`.
3. Descarga `datos/color/` desde el enlace de arriba y colócala junto al
   notebook antes de correr la sección de verificación de labels.
4. Corre las celdas en orden — la primera instala todas las dependencias, y el
   modelo base se importa directo desde el Hub (`Qwen/Qwen2.5-0.5B-Instruct`).

## Dataset

No existe un dataset público de recomendaciones agronómicas en español para las
etiquetas de PlantVillage — el dataset (`datos/dataset_finetuning_plantvillage.jsonl`)
fue curado por el equipo a partir de fuentes de extensión agrícola universitaria
(UMN, Penn State, MSU, UF/IFAS, Cornell, UC IPM, entre otras). Fuentes completas
en `datos/REFERENCIAS.md`; construcción, verificación contra el dataset real de
imágenes, y limitaciones documentadas en la sección 2 del notebook.

## Referencias principales

- Dataset: https://github.com/spmohanty/plantvillage-dataset
- Dataset de imágenes (solo las de color): https://huggingface.co/datasets/Jackieeeeee/plantvillage-raw-color-7f7ecc7
- Modelo base: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
