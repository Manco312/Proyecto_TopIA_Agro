# Declaración de uso de Inteligencia Artificial — Entrega M4

**Equipo:** Luciana Hoyos · Sara López · Juan Carlos Citelly · Santiago Manco Maya

---

## Herramienta utilizada

Se utilizó **Claude (Anthropic)** como asistente durante el desarrollo de esta entrega.
A continuación se detalla en qué tareas se usó, en cuáles no, y qué quedó bajo revisión y
decisión directa del equipo.

## Tareas en las que se usó IA

- **Código muy técnico o repetitivo del componente visual:** la función `ver()` sobre CLIP
  ViT-B/32 (zero-shot con las 38 clases escritas como texto y cabeza lineal sobre los rasgos
  congelados), las cuatro guardas de la válvula, la calibración de los umbrales por percentil
  del conjunto retenido y el guardado de los artefactos en `vision/`.
- **Integración con el sistema de M3 sin reescribirlo:** el cargador que ejecuta las celdas de
  código de `notebookM3.ipynb` (§0–§14) en un espacio de nombres propio, el corte de las celdas
  en sus demos para no repetir cómputo, y el enchufe `sistema_multimodal()` que inyecta la
  etiqueta del clasificador en el estado del agente de M3.
- **Preparación del eval set con imagen:** la selección determinista de las imágenes desde
  `datos/color/`, la degradación documentada de `img_07.jpg` (desenfoque y oscurecimiento) y la
  descarga con atribución de la imagen de roya del café desde Wikimedia Commons.
- **Documentación del código y redacción general:** comentarios del código, textos markdown del
  notebook, README y `eval_set/REFERENCIAS.md`. Claude redactó de manera profesional los
  textos, incluidas la lectura honesta y las conclusiones, a partir de los resultados de la
  corrida del equipo.
- **Depuración:** detección y corrección de errores durante la corrida. Entre ellos: la variable
  `_cafe` que quedaba sin definir al cortar un *smoke test* de M3, el `OSError 1455` de Windows
  (límite de confirmación agotado porque la traza de una excepción anterior mantenía vivos los
  modelos ya cargados) y el cambio de API de CLIP en `transformers` 5.x, donde
  `get_image_features` y `get_text_features` dejaron de devolver un tensor.

## Tareas que NO se delegaron a la IA

- **Definición del alcance y de las decisiones de diseño:** que el patrón fuera `tool` con un
  clasificador y no `input` con un VLM —decidido sobre la evidencia de M2 y M3, donde el cuello
  de botella medido está en la generación y no en el conocimiento—, que la interfaz fuera
  `consultar_ficha(etiqueta)`, que el sistema de M3 no se modificara y que los notebooks de M1,
  M2 y M3 se entregaran intactos.
- **El diseño del eval set y sus criterios:** qué siete casos, qué clase para cada uno y por
  qué. El caso de maíz se eligió porque la respuesta correcta es *no tratar*; el de vid porque
  es una clase ausente de los 27 casos de texto; el de café porque está en el catálogo del
  recomendador pero fuera de las 38 clases del clasificador. Los criterios se escribieron y se
  versionaron **antes** de correr nada.
- **La ejecución real del notebook:** correrlo de principio a fin, descargar los modelos,
  embeber las 3 040 imágenes y entrenar la cabeza lineal se hizo en el entorno del equipo
  (laptop con 6 GB de VRAM). Nada fue generado ni simulado por la IA. Los números del scorecard,
  de la calibración de la válvula y de la tabla de delta provienen de esa corrida.
- **La revisión de resultados y la lectura honesta:** el equipo revisó las salidas y decidió qué
  reportar y cómo. En particular: reportar por separado el aporte del encabezado determinista
  frente al del modelo; declarar que el fallo de `img-05` (recetar 50–100 g/ha sobre una hoja
  sana) es un fallo de seguridad del recomendador y no del componente visual, aunque la regla de
  dominio de M2 lo contara como acierto; y atribuir la diferencia de 9/22 a 12/22 en los casos
  de texto al cambio de versión de `torch` entre las dos corridas, en vez de apuntárselo a M4.
- **La elección de los modelos:** CLIP ViT-B/32 se evaluó y aceptó con base en las restricciones
  de la GPU del equipo (6 GB, con el sistema de M3 ya cargado) y en que deja el encoder
  congelado y disponible para M5. La decisión de quedarse con la cabeza lineal en vez del
  zero-shot se tomó sobre la medición de la §4 (0.914 frente a 0.172 de top-1), no por defecto.
