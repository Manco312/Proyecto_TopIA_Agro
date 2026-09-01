# Rúbrica del juez (LLM-as-a-judge) — Entrega M2

**Equipo:** Luciana Hoyos · Sara López · Juan Carlos Citelly · Santiago Manco Maya
**Dominio:** recomendador de manejo agronómico en español para las 38 clases de PlantVillage.

Esta rúbrica es **parte del entregable** y está versionada en el repo. El `notebook.ipynb`
la carga textualmente desde la constante `RUBRICA` (que reproduce este archivo) y también
la re-escribe a disco al final, para que la versión usada en una corrida quede registrada
junto al scorecard.

---

## Qué evalúa el juez

Dada una **pregunta de un agricultor** sobre una hoja enferma y la **respuesta del
sistema**, el juez asigna **un entero de 1 a 5** que mide qué tan buena es la respuesta
como recomendación agronómica. El juez recibe también la **respuesta de referencia** del
`eval_set` como guía (no para exigir coincidencia literal).

## Escala (anclas explícitas 1–5)

```
5 = Correcta y accionable. Identifica bien el problema/patógeno, la acción recomendada es
    agronómicamente correcta para ese caso (incluye "no tratar" cuando corresponde), y la
    prevención es pertinente. No inventa datos. Un agricultor podría actuar con esto.

4 = Correcta con un detalle menor mejorable: identificación y acción correctas, pero falta
    parte de la prevención, o hay una imprecisión secundaria que no cambia la decisión.

3 = Parcialmente correcta o incompleta: acierta el problema general pero la acción es vaga,
    genérica ("aplica un fungicida" sin más) o le falta un componente importante. Sirve a
    medias.

2 = Mayormente incorrecta o confusa: confunde el tipo de patógeno (trata un virus o una
    plaga como hongo), recomienda una acción inadecuada, o mezcla información de otra
    enfermedad. Engaña más de lo que ayuda.

1 = Incorrecta, irrelevante o inventada: no reconoce el dominio agronómico, alucina un
    patógeno o un tratamiento que no existe, o —en un caso adversarial— responde con
    seguridad algo que debía rechazar o corregir.
```

## Reglas de aplicación (para acotar el criterio del juez y sus sesgos)

- **La extensión NO es un criterio.** Una respuesta corta y correcta vale más que una
  larga y vaga. El juez tiene instrucción explícita de ignorar el largo del texto.
- **Casos adversariales:** si la pregunta tiene una premisa falsa, pide algo fuera del
  dominio (cultivo no PlantVillage) o pide una dosis exacta de agroquímico / periodo de
  carencia, la respuesta **buena rechaza o corrige**; aceptar la premisa y responder con
  seguridad es un **1–2**.
- **"No tratar" es una respuesta válida.** En clases como roya común del maíz o cultivos
  sanos, recomendar tratamiento innecesario **baja** la nota.
- **Formato esperado:** identificación → acción recomendada → prevención. No cumplirlo no
  es un 1 automático, pero sí baja de 5 a 4/3 según cuánto se pierda.
- **Salida del juez:** un solo dígito 1–5, sin explicación (para poder parsearlo). El
  parser toma el primer dígito 1–5 de la salida; si no hay, cae a 3 (neutro) y se registra.

## Prompt del juez (pointwise, reproducible)

- `system`: "Eres un evaluador agronómico estricto y objetivo. La extensión de la
  respuesta no es un criterio de calidad."
- `user`: la rúbrica de arriba + la pregunta + la respuesta a evaluar + la respuesta de
  referencia + "Responde SOLO con un dígito del 1 al 5. Sin explicación."
- Generación **determinista**: `do_sample=False`, `temperature` no aplica, `max_new_tokens=5`,
  `seed` global fijo.

## Modelos juez

| Rol | Modelo | Familia | Para qué |
|---|---|---|---|
| Juez principal | `Qwen/Qwen2.5-1.5B-Instruct` | Qwen2.5 | puntaje pointwise 1–5 del scorecard |
| Juez de control | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | SmolLM2 (otra familia) | medir **auto-preferencia**: el sistema evaluado es Qwen2.5-0.5B afinado, misma familia que el juez principal |

Ambos IDs y sus `revision` (commit hash) quedan impresos y guardados en el scorecard para
que otra persona reproduzca la corrida exacta.
