# Procedencia y licencia de las imágenes del eval set de M4

Las siete imágenes de `imagenes/` son de **dataset open source** o derivadas de uno. Ninguna es
propia, ninguna muestra personas, caras, documentos, matrículas ni ubicaciones identificables.
Ninguna fue generada con IA (este módulo usa un patrón de **clasificación**, no de difusión; por
eso no hay ninguna imagen con etiqueta "GENERADA").

Los nombres de archivo son deliberadamente neutros (`img_01.jpg` … `img_07.jpg`): **el nombre
original del dataset contiene la etiqueta de la clase**, y dejarlo habría sido una fuga de
etiqueta hacia cualquier parte del pipeline que leyera la ruta. La etiqueta verdadera vive solo
en la columna `etiqueta_real` de [`casos_m4.csv`](casos_m4.csv).

## 1 · Imágenes de PlantVillage (6 de 7)

**Dataset:** PlantVillage, subconjunto `color`.
**Fuente:** https://github.com/spmohanty/plantvillage-dataset ·
espejo usado por el equipo desde M1: https://huggingface.co/datasets/Jackieeeeee/plantvillage-raw-color-7f7ecc7
**Licencia:** CC0 1.0 (dominio público). Son fotografías de hojas sobre fondo uniforme tomadas en
condiciones de laboratorio; no contienen datos personales.
**Cita:** Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant
health to enable the development of mobile disease diagnostics.* arXiv:1511.08060.

Copiadas sin modificar desde `M4/datos/color/<clase>/` (carpeta ignorada por git: es el dataset
crudo de 54 000 imágenes, no se versiona).

| Archivo | Clase PlantVillage | Archivo original en el dataset |
|---|---|---|
| `img_01.jpg` | `Potato___Late_blight` | `01a8cc9f-074a-4866-87c8-bb5a9e3895b4___RS_LB 2968.JPG` |
| `img_02.jpg` | `Tomato___Leaf_Mold` | `036d12b0-1e97-4975-bd80-a8a3c48588e7___Crnl_L.Mold 6500.JPG` |
| `img_03.jpg` | `Corn_(maize)___Common_rust_` | `RS_Rust 1570.JPG` |
| `img_04.jpg` | `Grape___Esca_(Black_Measles)` | `01c74b8d-1113-47c0-86a3-6ceea27853c0___FAM_B.Msls 4113.JPG` |
| `img_05.jpg` | `Tomato___healthy` | `0172e56c-8bb8-4e75-8ac7-509df81393e8___RS_HL 0580.JPG` |

### `img_07.jpg` — derivada, degradada a propósito

Misma fuente y licencia (PlantVillage, CC0), pero **transformada** para construir el caso
adversarial de foto inutilizable:

- **Original:** `Potato___Late_blight / 08ddae0c-5328-4183-b16e-c980f197e471___RS_LB 2845.JPG`
  (una imagen **distinta** a la de `img_01.jpg`, para que no sea el mismo archivo dos veces).
- **Transformación exacta:** `GaussianBlur(radius=3.2)` → `Brightness × 0.38` → `Contrast × 0.70`,
  con Pillow, guardada a `quality=88`.
- **Por qué:** mide si la válvula del clasificador se dispara cuando la imagen no da señal
  suficiente, en vez de adivinar una de las 38 clases con confianza alta.

## 2 · `img_06.jpg` — roya del café (fuera de las 38 clases)

**Archivo:** *Hemileia vastatrix - coffee leaf rust.jpg*
**Autor:** Smartse (Wikimedia Commons)
**Licencia:** **CC BY-SA 3.0** — https://creativecommons.org/licenses/by-sa/3.0
**Página del archivo:** https://commons.wikimedia.org/wiki/File:Hemileia_vastatrix_-_coffee_leaf_rust.jpg
**Modificación:** redimensionada a 640 px de lado mayor y re-guardada como JPEG `quality=92`.
Sin otros cambios.

> Atribución requerida por la licencia: *"Hemileia vastatrix - coffee leaf rust"* por Smartse,
> CC BY-SA 3.0, vía Wikimedia Commons. Redimensionada.

**Por qué esta imagen y no otra de PlantVillage:** el café **no está entre las 38 clases** del
clasificador visual, pero **sí está en el catálogo del recomendador** (ficha de Cenicafé que M3
incorporó en `datos/pdfs/externo/ica-11-cafe-roya.pdf`). Es el único caso del eval set donde el
componente visual *no puede* acertar por construcción, y por eso mide exactamente lo que debe
medir: que la válvula se dispare y el sistema de M3 siga respondiendo por texto, en vez de que
el clasificador invente una de las 38 etiquetas que sí conoce. Enlaza directamente con el caso
`adv-01-alucinacion-cafe` del eval set de texto de M2/M3.

**Qué pasó en la corrida:** el clasificador propuso `Potato___Early_blight` —exactamente la
alucinación que el caso buscaba provocar— y **tres guardas independientes** lo detuvieron:
confianza 0.211 (umbral 0.217), distancia al prototipo 0.86 (umbral 0.922) y conflicto de
cultivo (la pregunta dice "cafetal", la foto se parece a papa). El sistema no nombró ninguna de
las 38 enfermedades y respondió por texto desde la ficha de Cenicafé. Detalle en la §9 del
notebook.

## 3 · Cómo se reconstruyen

`imagenes/_procedencia.json` guarda, en formato legible por máquina, la clase y el archivo
original de cada imagen derivada de PlantVillage, junto con la transformación aplicada. Con el
dataset `color` descargado, el mapa de esa tabla reproduce las seis imágenes de PlantVillage
byte a byte salvo la recompresión JPEG; `img_06.jpg` se descarga de la URL de Commons de arriba.
