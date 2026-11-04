# Declaración de uso de Inteligencia Artificial

**Equipo:** Luciana Hoyos · Sara López · Juan Carlos Citelly · Santiago Manco Maya

---

## Herramienta utilizada

Se utilizó **Claude (Anthropic)** como asistente durante el desarrollo de esta
entrega. A continuación se detalla en qué tareas se usó, en cuáles
no, y qué quedó bajo revisión y decisión directa del equipo.

## Tareas en las que se usó IA

- **Trabajo inicial del notebook:** código base para
  cargar el dataset, hacer el split train/val/test, configurar LoRA
  y evaluar con ROUGE, siguiendo el formato del laboratorio de la clase.
- **Depuración de errores técnicos:** resolución de incompatibilidades de
  versiones de librerías (`trl`, `SFTConfig`), errores de `train_test_split`
  con clases de pocos ejemplos, y ajustes de API que cambiaron entre versiones.
- **Redacción de una primera versión de las recomendaciones agronómicas** de la
  base de conocimiento (`base_conocimiento_plantvillage.json`), a partir de
  fuentes primarias de extensión agrícola universitaria que la IA identificó y
  citó explícitamente (UMN Extension, Penn State, MSU, UF/IFAS, Cornell, UC IPM,
  entre otras — ver `datos/REFERENCIAS.md`).
- **Detección y corrección de discrepancias** entre los nombres de clase usados
  en el dataset propio y los nombres reales de las carpetas del dataset
  PlantVillage descargado.
- **Redacción formal de la documentación:** Apoyó en la REDACCIÓN de los borradores del README y secciones de
  markdown del notebook.

## Tareas que NO se delegaron a la IA

- **La ejecución real del entrenamiento y la evaluación** (correr las celdas,
  obtener las métricas de ROUGE, revisar las curvas de loss) se hizo en el
  entorno del equipo, no fue generada ni simulada por la IA.
- **La lectura, interpretación y validación de los resultados numéricos**
  (por ejemplo, notar que el validation loss quedó por debajo del training loss,
  o identificar el caso de alucinación del modelo fine-tuned con una pregunta
  fuera de distribución) fue discutida por el equipo, con apoyo de
  la IA para redactar esa lectura de forma clara, no para decidir qué decir.
- **La verificación del contenido agronómico** frente a las fuentes citadas fue
  revisión del equipo.
- **La selección final de hiperparámetros y del modelo base** fue evaluada y
  aceptada por el equipo con base en la evidencia presentada, no aplicada sin
  revisión.
