# Referencias del dataset de enfermedades foliares (PlantVillage)

## Sobre este dataset

No existe un dataset público de "recomendaciones agronómicas en español para las
38 etiquetas de PlantVillage". Este dataset es **derivado y curado**: la taxonomía
de las 38 clases viene del dataset original de imágenes (PlantVillage), pero el
contenido de identificación/tratamiento/prevención de cada clase fue redactado a
partir de las fuentes listadas abajo — principalmente servicios de extensión
agrícola universitaria de EE. UU. (Land-Grant Universities), que son fuentes
primarias abiertas, con revisión de especialistas en fitopatología, y son las
que usan agricultores y agrónomos reales para tomar decisiones de manejo.

## Origen de la taxonomía de clases (38 clases, 14 especies)

- Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on
  plant health to enable the development of mobile disease diagnostics*.
  arXiv:1511.08060. https://arxiv.org/abs/1511.08060
- Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using Deep Learning for
  Image-Based Plant Disease Detection*. Frontiers in Plant Science, 7:1419.
  https://doi.org/10.3389/fpls.2016.01419

## Fuentes de manejo agronómico por enfermedad (servicios de extensión)

- UMN Extension (University of Minnesota) — múltiples fichas de enfermedades:
  https://extension.umn.edu/plant-diseases/apple-scab
  https://extension.umn.edu/plant-diseases/cedar-apple-rust
  https://extension.umn.edu/corn-pest-management/gray-leaf-spot-corn
  https://extension.umn.edu/disease-management/bacterial-spot-tomato-and-pepper
  https://extension.umn.edu/disease-management/tomato-viruses
- Wisconsin Horticulture (UW-Madison), "Apple Scab":
  https://hort.extension.wisc.edu/articles/applescab/
- Penn State Extension, "Grape Disease - Black Rot":
  https://extension.psu.edu/grape-disease-black-rot
- WSU Tree Fruit (Washington State University), "Cherry Powdery Mildew":
  https://treefruit.wsu.edu/crop-protection/disease-management/cherry-powdery-mildew/
- NC State Extension, "Gray Leaf Spot in Corn":
  https://content.ces.ncsu.edu/gray-leaf-spot-in-corn
- Golden Harvest / Purdue University, "Northern Corn Leaf Blight and Gray Leaf
  Spot Corn": https://www.goldenharvestseeds.com/agronomy/articles/northern-corn-leaf-blight-and-gray-leaf-spot
- University of Arkansas Extension (UAEX), "Corn Diseases":
  https://uaex.uada.edu/farm-ranch/pest-management/plant-disease/corn.aspx
- MSU Extension (Michigan State University):
  https://www.canr.msu.edu/news/managing_black_rot_on_grapes
  https://www.canr.msu.edu/news/management_of_bacterial_spot_on_peach_and_nectarines
- Florida A&M University CAFS, "Post-Harvest Vineyard Management":
  https://cafs.famu.edu/departments-and-centers/research/center-for-viticulture-and-small-fruit-research/pdf/Post-Harvest-Vineyard-Management-fact-sheet.pdf
- UF/IFAS Extension (University of Florida):
  https://ask.ifas.ufl.edu/topics/citrus-greening
  https://ask.ifas.ufl.edu/publication/PP359
- UC Riverside CISR, "Huanglongbing (HLB or Citrus Greening)":
  https://cisr.ucr.edu/invasive-species/huanglongbing-hlb-or-citrus-greening
- UConn IPM (University of Connecticut), "Managing Bacterial Leaf Spot":
  https://ipm.cahnr.uconn.edu/managing-bacterial-leaf-spot/
- University of Kentucky, "Scouting Guides for Problems of Vegetables":
  https://veggiescout.mgcafe.uky.edu/node/175
- UW-Madison Vegetable Pathology (Dept. of Plant Pathology):
  https://vegpath.plantpath.wisc.edu/tag/late-blight-fungicides
- Cornell University (M. T. McGrath), "Late Blight Occurrence and Management in
  Potatoes and Tomatoes in the Northeastern United States":
  https://portage.extension.wisc.edu/files/2010/05/LateBlightOccurrenceandMGT2009McGrath.pdf
- Utah State University Extension, "Powdery Mildew on Vegetables":
  https://extension.usu.edu/planthealth/ipm/notes_ag/veg-powdery-mildew
- UC IPM (University of California), "Leaf Scorch of Strawberries":
  https://ipm.ucanr.edu/home-and-landscape/leaf-scorch-of-strawberries/
- UMass Extension, "Small Fruit IPM Fact Sheet SB-003":
  https://www.umass.edu/agriculture-food-environment/sites/ag.umass.edu/files/fact-sheets/pdf/strawberry_foliar_disease_fact_sheet_sb-003_0.pdf
- University of Maryland Extension, "Key to Common Problems of Tomatoes":
  https://extension.umd.edu/resource/key-common-problems-tomatoes

## Limitaciones a documentar en el proyecto

- Las recomendaciones son de carácter general/educativo: no sustituyen la
  evaluación de un agrónomo en campo ni indican dosis exactas de agroquímicos
  (por eso se citan principios activos o categorías, no dosis).
- La mayoría de las fuentes son de EE. UU.; el manejo real (productos
  registrados, dosis, clima) varía por país y región. Para uso en Colombia,
  estas recomendaciones deberían contrastarse con guías de ICA/Corpoica o
  agremiaciones locales del cultivo correspondiente.
- Para 12 de las 38 clases (los healthy) no hay "tratamiento" real, solo buenas
  prácticas; se documentan así explícitamente en el campo "tratamiento".
- Algunas entradas (p. ej. Apple___healthy, Corn___healthy) son una síntesis de
  buenas prácticas generales más que una cita textual de una fuente puntual.
