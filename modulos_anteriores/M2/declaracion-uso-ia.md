# Declaración de uso de Inteligencia Artificial — Entrega M2

**Equipo:** Luciana Hoyos · Sara López · Juan Carlos Citelly · Santiago Manco Maya

---

## Herramienta utilizada

Se utilizó **Claude (Anthropic)** como asistente durante el desarrollo de esta entrega. 
A continuación se detalla en qué tareas se
usó, en cuáles no, y qué quedó bajo revisión y decisión directa del equipo.

## Tareas en las que se usó IA

- **Implementación de las mitigaciones de sesgo del juez:** comparación *pairwise* en
  ambos órdenes (posición), instrucción anti-longitud en la rúbrica + correlación de
  Spearman largo↔nota (longitud), y el segundo juez de otra familia (`SmolLM2-1.7B`) para
  medir auto-preferencia.
- **Redacción general:** En esta entrega, Claude se encargó de redactar de manera profesional
los textos que el equipo había planteado, para asegurar así una mejor calidad en la entrega.
- **Depuración de la lógica de "aciertos de dominio":** detección y corrección de falsos
  positivos en el detector de abstención (p. ej. que la subcadena "ica" hacía *match*
  dentro de "aplica" o "identificación").

## Tareas que NO se delegaron a la IA

- **Estructuración base del notebook:** El equipo estructura el notebook tanto con sus secciones
de código como con sus secciones de markdown.
- **La ejecución real del harness:** (correr el notebook de principio a fin, descargar los
  modelos, obtener el scorecard del baseline y las métricas de sesgo del juez) se hace en
  el entorno del equipo, no fue generada ni simulada por la IA.
  Los números del scorecard en el repo provienen de esa corrida.
- **La lectura honesta del scorecard:** La interpretación final —qué debilidad
  concreta revela cada número, y contrastarla con el detalle caso por caso— es en base a
  discusión del equipo.
- **La construcción del eval set:** El equipo construye el eval set, con sus respectivos
gold y casos adversariales.
- **La elección de los modelos juez** (Qwen2.5-1.5B como principal, SmolLM2-1.7B como
  control de otra familia) fue evaluada y aceptada por el equipo con base en las
  restricciones de la GPU y en la necesidad de un juez fuera de la familia del
  sistema evaluado.
