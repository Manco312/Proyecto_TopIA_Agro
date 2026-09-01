# Declaración de uso de Inteligencia Artificial — Entrega M2

**Equipo:** Luciana Hoyos · Sara López · Juan Carlos Citelly · Santiago Manco Maya

---

## Herramienta utilizada

Se utilizó **Claude (Anthropic)** como asistente durante el desarrollo de esta entrega
(el harness de evaluación de 3 dimensiones). A continuación se detalla en qué tareas se
usó, en cuáles no, y qué quedó bajo revisión y decisión directa del equipo.

## Tareas en las que se usó IA

- **Estructura del `notebook.ipynb` del harness:** código base de las tres dimensiones
  (similitud por embeddings + ROUGE-L, LLM-as-a-judge pointwise con rúbrica 1–5, y la
  regla de "aciertos de dominio"), siguiendo el formato de los labs S05 y S06 de la clase.
- **Implementación de las mitigaciones de sesgo del juez:** comparación *pairwise* en
  ambos órdenes (posición), instrucción anti-longitud en la rúbrica + correlación de
  Spearman largo↔nota (longitud), y el segundo juez de otra familia (`SmolLM2-1.7B`) para
  medir auto-preferencia.
- **Redacción de una primera versión del eval set (`eval_set.json`):** los 13 casos
  (10 *gold* + 3 adversariales) se redactaron a partir de la base de conocimiento de M1
  (`datos/base_conocimiento_plantvillage.json`) y sus fuentes de extensión agrícola. El
  equipo eligió qué clases cubrir, revisó que cada `esperado` fuera fiel a la fuente y
  diseñó los tres casos adversariales (café fuera de dominio, premisa falsa "la roña es un
  virus", y la petición de dosis exacta de paraquat).
- **Redacción de la rúbrica del juez (`RUBRICA.md`):** primera versión de las anclas 1–5 y
  de las reglas anti-sesgo; el equipo ajustó los criterios agronómicos (p. ej. que
  "no tratar" sea una respuesta válida).
- **Redacción formal de la documentación:** borradores del `README.md`, `RUBRICA.md` y las
  celdas markdown del notebook.
- **Depuración de la lógica de "aciertos de dominio":** detección y corrección de falsos
  positivos en el detector de abstención (p. ej. que la subcadena "ica" hacía *match*
  dentro de "aplica" o "identificación").

## Tareas que NO se delegaron a la IA

- **La ejecución real del harness** (correr el notebook de principio a fin, descargar los
  modelos, obtener el scorecard del baseline y las métricas de sesgo del juez) se hace en
  el entorno del equipo (Google Colab con GPU T4), no fue generada ni simulada por la IA.
  Los números del scorecard en el repo provienen de esa corrida.
- **La lectura honesta del scorecard** (§8 del notebook): el notebook imprime un *borrador*
  automático a partir de los números reales, pero la interpretación final —qué debilidad
  concreta revela cada número, y contrastarla con el detalle caso por caso— es discusión y
  redacción del equipo.
- **La validación del contenido agronómico** de cada `esperado` frente a las fuentes
  citadas fue revisión del equipo.
- **La decisión de qué casos adversariales incluir** y qué cuenta como "abstención
  correcta" en cada uno fue del equipo; la IA solo implementó la regla acordada.
- **La elección de los modelos juez** (Qwen2.5-1.5B como principal, SmolLM2-1.7B como
  control de otra familia) fue evaluada y aceptada por el equipo con base en las
  restricciones de la GPU de Colab y en la necesidad de un juez fuera de la familia del
  sistema evaluado.
