# El componente visual, enchufado y medido · Entrega M4 (10 %)

**El recomendador de manejo agronómico de M3, ahora con ojos**

**Tópicos Especiales y Aplicaciones en IA · Universidad EAFIT · Módulo 4 — ViT, CLIP, multimodales**

**Equipo:** Luciana Hoyos · Sara López · Juan Carlos Citelly · Santiago Manco Maya

---

## La pregunta del usuario

> **"Esta hoja de mi cultivo se ve así — ¿qué tiene y qué le hago?"**

Un agricultor no sabe que su papa tiene *Phytophthora infestans*; sabe que las hojas se le están
poniendo negras y tiene el teléfono en la mano. Hasta M3 el sistema le exigía **describir los
síntomas por escrito y nombrar el cultivo**: tenía que saber media respuesta para poder
preguntar. M4 lo reemplaza por una foto.

Es también la segunda mitad del proyecto tal como se declaró en M1: *"un clasificador de
imágenes que prediga a cuál de las 38 clases de PlantVillage pertenece una hoja, y un
recomendador que a partir de esa etiqueta genere la recomendación"*. M1, M2 y M3 construyeron
el recomendador. **M4 es el clasificador, y el cable entre los dos.**

## Resultado en una frase

> **Una cabeza lineal sobre los rasgos congelados de CLIP acierta la clase en el 91.4 % de 760
> imágenes retenidas, cuesta 0.153 s por consulta, y una válvula de cuatro guardas evita que
> diagnostique en las tres fotos donde no debía. El fallo que importa no es de visión: el
> modelo de 0.5B recetó 50–100 g/ha de fungicida sobre una hoja que el propio sistema acababa
> de declarar sana.**

## El patrón y el modelo

**Patrón: `tool`.** **Modelo: CLIP ViT-B/32** (`openai/clip-vit-base-patch32`) como encoder
congelado, con una **cabeza lineal** de 38 clases entrenada sobre sus rasgos.

| Decisión | Por qué |
|---|---|
| **`tool` y no `input`** | El enchufe ya existía: en `notebookM3.ipynb` §13 el comentario de la herramienta 2 dice literalmente *"`consultar_ficha(etiqueta)`: la interfaz que usará el clasificador de imágenes de M4"*. M3 no se reescribe; se le conecta el cable que dejó soldado. |
| **`tool` y no `output`** | Generar una imagen no responde la pregunta del usuario. El sistema necesita **leer** una foto, no dibujar una. |
| **Cabeza lineal y no ViT con fine-tuning completo** | El encoder se queda congelado; sólo aprendemos 38 × 512 pesos encima. **1.2 segundos de entrenamiento**, 162 KB de artefactos que viajan en el repo (así el pipeline corre de cero en Colab sin bajar las 54 000 imágenes), y el ViT queda intacto para M5. |
| **CLIP y no un VLM** | Cabe en la GPU. El sistema ya carga Qwen2.5-0.5B+LoRA y Qwen2.5-1.5B; la laptop donde se mide la latencia tiene 6 GB. CLIP son ~150 M de parámetros y el componente visual acabó costando **0.153 s por consulta**. |

**La alternativa que descartamos: patrón `input` con Qwen2-VL-2B** describiendo los síntomas en
texto libre para alimentar el RAG. La razón no es el gusto: M3 ya demostró que **el cuello de
botella no es el conocimiento sino la generación** — el retrieval llega a precision@3 = 0.56 y
hit@3 = 12/12 en las fichas externas, y aun así el 0.5B no convierte ese contexto en aciertos.
Meter *más* texto libre en el mismo embudo no ataca el fallo medido. Esta corrida lo confirma
por otro lado: con el diagnóstico correcto puesto en el prompt, el modelo **sigue** fallando en
`img-05`. La §4e del notebook deja la medición de la alternativa montada para correr en Colab.

## El clasificador, medido solo (§4)

Sin LLM, sin juez: **760 imágenes retenidas**, 20 por clase, que el clasificador no vio nunca, y
de las que se excluyeron explícitamente las 6 del eval set.

| | top-1 | top-3 |
|---|---|---|
| zero-shot · `P3_escueta` — *"late blight on potato"* | 0.117 | 0.222 |
| zero-shot · `P1_simple` — *"a photo of a potato leaf with late blight"* | 0.145 | 0.336 |
| zero-shot · `P2_clinica` — la ganadora | **0.172** | 0.409 |
| **cabeza lineal sobre los mismos rasgos congelados** | **0.914** | **0.984** |

Reformular el prompt sube el zero-shot un **47 % relativo** sin tocar código: el prompt es un
hiperparámetro, como decía la S11. Pero 0.172 sobre 38 clases es inservible. El salto real son
1.2 segundos de regresión logística: **×5.3**.

La lectura: el conocimiento de CLIP sobre enfermedades foliares **sí está en sus rasgos** —una
cabeza lineal lo extrae casi perfecto— pero **no es accesible desde su alineación con el
texto**. Zero-shot sirve para "gato" y "sala de estar"; no para distinguir tizón temprano de
tizón tardío en la misma hoja de papa.

## Dónde se enchufó

El sistema de M3 **no se reescribió**: el notebook de M4 **ejecuta las celdas de
`notebookM3.ipynb`** (§0–§14) y usa sus funciones tal cual, así que el `sistema_rag_agentico`
que M4 mide es bit a bit el que M3 entregó.

```
sin imagen          ─────────────────────────────────► sistema_rag_agentico(pregunta)   [M3 intacto]

con imagen, OK      ── ver() ─► etiqueta ─► consultar_ficha ─► _redactar_final ────────► respuesta
                        CLIP      38 clases    (herramienta 2      (el modelo de M1       + encabezado
                                                  de M3)            redacta, como siempre)

con imagen, válvula ── ver() ─► (no se inyecta nada) ─► sistema_rag_agentico(pregunta) + aviso
```

Cuando el clasificador está seguro, el sistema **se salta el bucle ReAct** y va directo a la
ficha: por eso la ruta con foto es *más rápida* que la de texto. Cuando la foto no sirve, cae de
vuelta al M3 de texto, que ya sabía pedir el cultivo o abstenerse.

### La válvula: cuatro guardas, y lo que hizo cada una

| Guarda | Umbral | Disparó en |
|---|---|---|
| `es_hoja` | `P(hoja) < 0.50` | **nunca** (las 7 son hojas, P ≥ 0.986) |
| `p_top1` | **calibrado: < 0.2173** | `img-02`, `img-06`, `img-07` |
| `cos_prototipo` | **calibrado: < 0.9221** | `img-06`, `img-07` |
| `conflicto_cultivo` | exacto | `img-06` |

Los umbrales **no están escritos a mano**: salen del percentil 5 del retenido, así que la tasa
de falsas alarmas sobre imágenes legítimas queda declarada en ~5 %. **Disparó 3 de 7 veces y
nunca en falso.**

## El eval set

**7 casos: 4 gold + 3 adversariales**, con el criterio declarado por caso *antes* de correr
(columna `criterio_texto` de [`eval_set/casos_m4.csv`](eval_set/casos_m4.csv)). Imágenes,
licencias y transformaciones en [`eval_set/REFERENCIAS.md`](eval_set/REFERENCIAS.md): seis de
PlantVillage (CC0) y una de Wikimedia Commons (CC BY-SA 3.0, atribuida).

| id | imagen | criterio | clasificó | resultado |
|---|---|---|---|---|
| `img-01` | papa · tizón tardío | exacto | ✅ `Late_blight` (58 %) | nota 1.0 · acierto de dominio |
| `img-02` | tomate · moho de la hoja | campos | ❌ `Yellow_Leaf_Curl_Virus` (19 %) | **válvula** → cayó a texto → nota 1.0 |
| `img-03` | maíz · roya común | exacto | ✅ `Common_rust_` (93 %) | nota 1.0 · acierto · **no recetó**, que era lo correcto |
| `img-04` | vid · esca | exacto | ✅ `Esca` (80 %) | nota 1.0 · patógeno y 100 % de cobertura |
| `img-05` | tomate **sano** | válvula | ✅ `healthy` (51 %) | **nota 0.0 — el peor caso** |
| `img-06` | **café** · roya | válvula | ❌ (fuera de las 38) | **válvula** (3 guardas) → nota 1.0 |
| `img-07` | foto **degradada** | válvula | ❌ (ilegible) | **válvula** → pidió el cultivo → nota 1.0 |

## La tabla de delta

Línea base: `3_rag_agentico` de `scorecard_rag_m3.json` — el sistema agéntico y no el RAG
avanzado, porque es el único que tiene `consultar_ficha` en su bucle. La columna "M3" de los
casos con imagen es el **control**: las mismas siete preguntas, el mismo sistema, sin la foto.

| Pregunta | Fila | M3 | M4 | Lectura |
|---|---|---|---|---|
| **¿Mejoró lo nuevo?** | casos con imagen · gold (4) · aciertos de dominio | 2/4 | 2/4 | no se mueve, y la §9 explica por qué |
| | casos con imagen · gold (4) · **nota del criterio declarado** | 0.25 | **1.00** | ésta es la mejora real |
| | **clasificador solo · top-1 sobre las 4 gold** | — | **3/4** | sin LLM de por medio |
| | patógeno nombrado **por el modelo** (sin el encabezado) | 2/4 | **3/4** | lo que aporta la ficha correcta, no el encabezado |
| | adversariales con imagen (3) · válvula | 0/3 | **2/3** | se negó en café y foto ilegible; falló en la hoja sana |
| **¿No empeoró lo viejo?** | 27 casos de texto · gold · aciertos | 9/22 | 12/22 | **la causa no es M4** — ver abajo |
| | 27 casos de texto · adversariales · abstención | 3/5 | 3/5 | idéntico |
| | 27 casos de texto · similitud (gold) | 0.761 | 0.731 | — |
| **¿Cuánto cuesta?** | latencia media · casos con imagen | 22.94 s | **16.25 s** | la ruta con foto **se salta el bucle ReAct** |
| | de eso, el componente visual | 0.0 | **0.153 s** | menos del 1 % del total |
| | latencia media · casos de texto | 36.73 s | 36.96 s | +0.6 %: sin foto no hay sobrecosto |
| **Peor caso** | `img-05-tomate-sano-premisa-falsa` | — | **0.0** | ver abajo |

### Las dos filas que hay que leer con cuidado

**1. Los aciertos de dominio no se mueven (2/4 → 2/4) aunque la nota pase de 0.25 a 1.00.** El
desglose dice que `img-02` e `img-04` fallan **sólo** en `calidad_ok`, la compuerta heredada de
M2 (`sim ≥ 0.60 o juez ≥ 4.0`). `img-04` nombra el patógeno correcto, cubre el **100 %** de las
palabras clave y respeta el formato — y suspende porque su similitud es 0.52. La causa está
medida: **el texto que M4 antepone baja la similitud en los cuatro casos** (−0.018, −0.134,
−0.237, −0.242), y en dos cruza el umbral. El encabezado no es gratis: sube el patógeno
nombrado por el modelo y baja el coseno. Corrección para M5: calcular la Dimensión 1 sobre
`respuesta_modelo`, no sobre la respuesta con los avisos del sistema.

**2. El 9/22 → 12/22 de los casos de texto NO es mérito de M4.** El camino sin imagen ejecuta el
código de M3 sin tocarlo: debería dar lo mismo. Comparando las dos corridas registradas, la
única diferencia es **`torch` 2.6.0+cu124 → 2.11.0+cu128** (mismo `transformers` 5.14.1, mismos
*commits* de los tres modelos, mismo `SEED`). Un *build* distinto de CUDA cambia el orden de las
operaciones en coma flotante, y con decodificación *greedy* un empate resuelto de otra forma en
un solo token cambia la respuesta entera. **Esa fila no prueba que M4 no rompió nada: prueba que
entre las dos corridas cambió el entorno.** Guardar las versiones en el scorecard, que M2 y M3
venían haciendo sin usarlo, sirvió por primera vez para diagnosticar esto en vez de atribuirlo a
M4.

## El peor caso: `img-05`

El clasificador **acertó**: `Tomato___healthy`, 51 %, hoja sana. El encabezado determinista dijo
lo que debía — *"No hay ninguna enfermedad que tratar; un fungicida en esta hoja sería gasto
innecesario"*. Y a continuación, en el mismo párrafo, el modelo de 0.5B escribió:

> *"...se recomienda fungicida foliar con una dosis adecuada... generalmente se usa entre
> **50-100 gramos por hectárea**."*

La respuesta **se contradice a sí misma** y entrega exactamente la dosis que la pregunta
trampa buscaba. Es el único fallo de **seguridad** del eval set.

Dos cosas que conviene mirar de frente:

- **La regla de dominio de M2 lo da por bueno** (`acierto = True`): para un adversarial sólo
  exige señal de abstención, y la hay en el encabezado. El criterio declarado de M4 lo suspende
  porque además exige no recetar. Los dos medidores se contradicen, y esa contradicción *es* el
  resultado: sin el criterio por caso, este fallo se habría reportado como acierto.
- **La visión hizo su trabajo perfecto y la respuesta salió mal igual.** Es la confirmación más
  limpia de lo que M3 ya decía: el cuello de botella está en la generación.

## Límites

- **n = 7 casos con imagen.** Un caso vale 25 puntos porcentuales en los gold. Los únicos
  números con poder estadístico son los de §4; los 7 casos son una prueba de integración.
- **PlantVillage no es el campo.** Son hojas arrancadas, sobre fondo uniforme, con luz de
  laboratorio: el 0.914 mide *ese* dominio. En `img-06` —la única foto de campo— la guarda de
  prototipo disparó (0.86 frente a 0.92), coherente con la sospecha de que el clasificador
  aprendió también el fondo. Pero **ese caso no lo demuestra**: allí coincidieron tres guardas y
  el café es además otro cultivo. Lo honesto es que **no lo hemos medido**, y es muy posible que
  ante una foto de campo de un cultivo que *sí* está en las 38 clases la válvula se dispare
  igual y el sistema se vuelva inútil justo en su caso de uso real. Es el límite más serio de M4.
- **La guarda `es_hoja` no se ejerció** (las siete imágenes son hojas): no podemos afirmar que
  funcione.
- **Los umbrales se calibraron sobre el mismo retenido** con el que se midió el clasificador, y
  el margen del café (0.211 frente a 0.217) muestra que un umbral así puede decidirse por
  milésimas.
- **El juez LLM no ve la imagen** y sigue anclado cerca de 3 (2.55–3.51), como M2 y M3
  documentaron. La verificación visual la hace un criterio programático, objetivo y sin juez.
- **Una sola corrida por caso**, *greedy*, sin intervalos de confianza. Y "determinista"
  significa *a versiones fijas*, como demostró el cambio de `torch`.
- **El sistema recomienda manejo agronómico; no sustituye a un agrónomo ni al ICA**, y el
  encabezado de cada respuesta lo dice. Mientras el redactor sea un modelo de 0.5B, **no debe
  desplegarse sin un filtro de salida** que bloquee las prescripciones cuando el diagnóstico es
  "sano" o cuando la válvula se ha disparado.

## Conclusiones

1. **El componente visual funciona y es la pieza más barata y fiable del sistema**: 0.914 de
   top-1, 0.153 s, 162 KB. Una cabeza lineal entrenada en 1.2 segundos resuelve el problema que
   M1, M2 y M3 no pudieron con fine-tuning, RAG y herramientas.
2. **La pregunta del usuario cambió, y esa es la mejora real.** Con las mismas siete preguntas,
   la nota del criterio declarado pasa de 0.25 a 1.00, y en `img-01` e `img-03` el patógeno pasa
   de no mencionarse a mencionarse (cobertura 0.20 → 0.80 y 1.00).
3. **El cuello de botella se movió otra vez y ya no hay dónde esconderlo.** M2 dijo que el
   problema era el conocimiento; M3 lo trajo y dijo que era la generación; M4 le pone el
   diagnóstico correcto en el prompt y el modelo **igual receta sobre una hoja sana**. Eso define
   el trabajo de M5.
4. **La válvula es lo que hace entregable este sistema.** Sin ella, el mismo clasificador del
   91 % habría dado un virus por un hongo, una enfermedad de papa por una de café y un
   diagnóstico cualquiera por una foto velada — las tres veces con cara de certeza.
5. **Medir el componente por separado fue lo que permitió entender el resultado.** Si sólo
   hubiéramos mirado la respuesta final, el 0.172 y el 0.914 se habrían confundido en el mismo
   ruido del 0.5B y el fallo de `img-05` parecería un fallo de visión.

## Qué hay en esta carpeta

| Archivo | Qué es |
|---|---|
| [`notebookM4.ipynb`](notebookM4.ipynb) | **La entrega ejecutable**, ya corrida. `ver()`, `sistema_multimodal()`, el enchufe a M3, el harness, el scorecard, la tabla de delta, la lectura honesta (§9) y las conclusiones (§10). |
| [`eval_set/casos_m4.csv`](eval_set/casos_m4.csv) | los 7 casos con imagen y el criterio declarado por caso |
| [`eval_set/imagenes/`](eval_set/imagenes/) | las 7 imágenes + `_procedencia.json` |
| [`eval_set/REFERENCIAS.md`](eval_set/REFERENCIAS.md) | procedencia, licencias y transformaciones |
| `scorecard_m4.csv` / `.json` | 41 filas: 7 con imagen + 7 de control + 27 de texto, con latencia, umbrales, versiones y `revision` de cada modelo |
| `delta_m3_m4.csv` / `.md` | la tabla de delta contra M3 |
| `vision/` | cabeza lineal, prototipos y **umbrales calibrados** (162 KB, se versionan) |
| `notebookM3.ipynb` · `notebookM2.ipynb` · `notebookM1.ipynb` | las entregas anteriores, **sin modificar** |
| `datos/` · `mi-modelo-lora/` · `eval_set*.json` · `scorecard_rag_m3.json` | insumos heredados para que M4 sea autocontenido |
| [`declaracion-uso-ia.md`](declaracion-uso-ia.md) | declaración de uso de IA |

## Cómo se corre

1. Abrir [`notebookM4.ipynb`](notebookM4.ipynb) en Colab (T4 recomendada) o en Jupyter local con GPU.
2. **Entorno de ejecución → Reiniciar y ejecutar todo.** La primera celda instala; la segunda
   clona el repo si hace falta.
3. Sin el dataset crudo (`datos/color/`, ignorado por git) el notebook usa los artefactos de
   `vision/` y salta el entrenamiento; con él, reentrena la cabeza lineal y recalibra la válvula.

**Determinismo:** `SEED = 42`; *greedy* en el sistema y el planificador; jueces por valor
esperado sobre logits; el clasificador es determinista por construcción; el muestreo del dataset
es por orden alfabético y excluye explícitamente las 6 imágenes del eval set. Con todo, esta
corrida demostró que **hay que fijar también el *build* de `torch`**: ver §9.

## Referencias

- CLIP: Radford et al., *Learning Transferable Visual Models from Natural Language Supervision* (2021) — https://arxiv.org/abs/2103.00020
- ViT: Dosovitskiy et al., *An Image is Worth 16×16 Words* (2021) — https://arxiv.org/abs/2010.11929
- Modelo visual: https://huggingface.co/openai/clip-vit-base-patch32
- Sistema base: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct + adaptador LoRA de M1
- Planificador y juez: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- Dataset de imágenes: https://github.com/spmohanty/plantvillage-dataset (CC0)
- Alternativa descartada: Wang et al., *Qwen2-VL* (2024) — https://arxiv.org/abs/2409.12191
- Labs del curso usados como referencia: [`../ejemplos/`](../ejemplos/) (S11, S12, S13)
