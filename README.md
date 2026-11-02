# Sistema de detección de alteraciones en imágenes

Procesamiento Digital de Imágenes (11804) - Universidad Nacional de Luján - 2026, 2do cuatrimestre

Integrantes: Thiago Puyelli, Matias Herrneder, Valentín J. Romero Monteagudo
Profesora: Lic. Sandra Serafino

Detección de **manipulación clásica de imágenes** (splicing, copy-move, eliminación
de objetos por inpainting). El sistema combina dos modelos:

1. **TruFor** (Guillaro et al., CVPR 2023, GRIP-UNINA), preentrenado y **congelado**.
   Por cada imagen devuelve un *score* global, un **mapa de manipulación** y un
   **mapa de confianza**.
2. Un **ViT propio, entrenado desde cero**, que recibe la imagen junto con los dos
   mapas de TruFor y devuelve la probabilidad de que la imagen esté manipulada.

El entrenamiento se hace sobre **IMD2020** (Kaggle: `saikotahmed/imd2020`) y está
pensado para correr en una **GPU T4 de Google Colab**.

```
imagen (tamaño variable)
   │  src/preprocess.py      variantes ANTES de TruFor (raw, jpeg70, blur1, ...)
   ▼
TruFor (congelado)  ──►  score + mapa + confianza  (resolución nativa)
   │  src/maps.py           NORMALIZACIÓN DE TAMAÑO: S×S + uint8
   ▼
cache  ──►  src/pack_cache.py  ──►  memmap local (N, 5, S, S)
   │
   ▼
ViT desde cero (RGB + mapa + confianza)  ──►  P(manipulada)
```

---

## 1. Requisitos

- **Git**, **curl** y **unzip**
- **uv** (gestor de entornos y paquetes de Python): <https://docs.astral.sh/uv/getting-started/installation/>
- GPU NVIDIA opcional para desarrollo. El entrenamiento serio va en Colab (T4).
- **Python 3.12**

## 2. Instalación

Todos los comandos se ejecutan **desde la raíz del repositorio**.

```bash
git clone https://github.com/Valentin190306/Deepfake_Detection_PDI_2026.git
cd Deepfake_Detection_PDI_2026

uv venv --python 3.12
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2.1 PyTorch (elegir UNA opción según el hardware)

**Sin GPU (solo CPU, versión liviana):**

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Con GPU NVIDIA:** tomar el comando exacto del selector de
<https://pytorch.org/get-started/locally/> y anteponerle `uv`:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/<cuXXX>
```

> En **Colab no se reinstala torch**: ya viene con CUDA y reinstalarlo suele romper la GPU.

### 2.2 Resto de dependencias

```bash
uv pip install -r requirements.txt
```

`timm` queda fijo en `0.9.16`: el código de TruFor usa la API vieja
(`timm.models.layers`). Con timm 1.x la carga del modelo puede romperse.

### 2.3 Bajar TruFor

El código de inferencia y los pesos (~270 MB) **no están en el repositorio**:

```bash
bash scripts/get_trufor.sh
```

Hace un *sparse checkout* de <https://github.com/grip-unina/TruFor> (solo
`test_docker/`, el paquete de inferencia autocontenido) y baja los pesos oficiales.
Es idempotente.

### 2.4 Verificar

```bash
python src/detect.py third_party/trufor/images/
```

Esperado: ~`0.19` para las *pristine* y ~`0.99` para las *tampered*.

---

## 3. Datos: IMD2020

### 3.1 Descarga

```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('saikotahmed/imd2020'))"
```

Si la descarga anónima falla, configurá `KAGGLE_USERNAME` / `KAGGLE_KEY`
(en Colab: *Secretos*).

### 3.2 Preparación

```bash
python src/prepare_imd2020.py --src <carpeta_kaggle> --out data --prefix imd_
```

Genera `data/images`, `data/masks` y **`data/manifest.csv`** con:

| columna | significado |
|---|---|
| `file`, `mask` | rutas relativas a `data/` (la máscara está vacía en auténticas) |
| `label` | 1 = manipulada, 0 = auténtica |
| `group` | **escena**: lo que usa el split para no filtrar datos |
| `source` | `tampered` / `authentic` |
| `width`, `height` | tamaño original |

**Sobre los grupos.** IMD2020 empareja cada manipulada con su original. Si el
original cae en train y su versión editada en validación, la métrica mide
memorización. Hay dos formas de definir la escena y **no son equivalentes**:

| `--group-by` | cuándo se usa | qué hace |
|---|---|---|
| `content` | default en layout plano | compara las **imágenes**, ignora el nombre |
| `name` | default si hay carpeta por escena | usa la carpeta o el nombre del archivo |
| `auto` | default | `content` en layout plano, `name` si hay carpeta por escena |

**Por qué el default es `content`.** En el repack de Kaggle, `img/` y
`Authentic/` usan el mismo esquema `<id>_orig` / `<id>_0`, pero sólo **38 de 414**
auténticas comparten id con una manipulada… y de esas 38, la mayoría ni siquiera
son la misma foto (correlación mediana 0.71, 29% por debajo de 0.5). Agrupar por
nombre ahí es agrupar al azar: medido sobre el dataset real, deja **186 de las
487 imágenes de validación (38%) con una foto gemela en train**.

Cómo funciona `content`:

1. **dHash** como filtro de candidatos (`--dhash-candidates`, default 16): barato
   y tolera la recompresión.
2. **Correlación normalizada (NCC)** como confirmación, calculada **ignorando la
   zona editada** (la máscara del manipulada, dilatada). Comparar la imagen
   entera castiga los pares con ediciones grandes: pares original/manipulada con
   NCC global 0.80–0.85 dan **0.995** fuera de la máscara. Con la zona editada
   ignorada, la misma foto da ~0.99 y dos fotos distintas 0.02 (control sobre
   IMD2020: p99.9 = 0.79, **0.06%** por encima de 0.90).
3. **complete-linkage**: dos grupos se unen sólo si **todos** sus pares superan
   `--ncc-threshold` (default 0.90). Single-linkage encadena y termina fusionando
   escenas distintas.
4. Sólo se comparan píxeles de imágenes del **mismo tamaño**: alinear
   resoluciones distintas exige reescalar y eso inventa parecidos.

En el dataset real eso da **1606 grupos** (contra 2131 por nombre), con **0
familias partidas** entre train y validación. Cuesta ~35 s y ~80 MB de RAM.

`--link-authentic` (tamaño + dHash mutuo) y `--dedupe-hash` siguen existiendo
pero **sólo aplican con `--group-by name`**: con `content` son innecesarios.

El emparejamiento imagen–máscara usa una clave distinta (`pair_key`, sólo saca
`_mask`), para que dos variantes de la misma escena no se pisen; esa misma clave
es el nombre de archivo en la salida (`imd_<pair_key>.jpg`,
`imd_<pair_key>__auth.jpg`), así cambiar la política de agrupamiento **no
recopia nada**.

Diagnostica además pares sin máscara, tamaños que no coinciden y máscaras
posiblemente invertidas (`--invert`).

> El dataset tiene **2010 manipuladas y 414 auténticas** (desbalance 5:1). El
> entrenamiento lo compensa con `pos_weight` y reporta balanced accuracy / AUC,
> no solo accuracy.

---

## 4. Mapas de TruFor: extracción y normalización

### 4.1 Por qué normalizar el tamaño

TruFor devuelve los mapas a **resolución nativa** (hasta ~3000×4400 en IMD2020).
Guardarlos así ocuparía decenas de GB, y el ViT necesita una entrada de tamaño
fijo. Por eso la normalización se hace **una sola vez, inmediatamente después de
TruFor**, y se cachea el resultado ya listo para entrenar.

### 4.2 Política de normalización (`src/maps.py`)

El presupuesto de píxeles es `S×S`: cualquier método descarta la misma cantidad de
información; lo que cambia es *cómo*.

- **Resize "squash" a S×S**, sin padding: no deforma nada relevante para una
  clasificación y no gasta píxeles en relleno.
- **Mapa de manipulación: `blend`** = `0.5·area + 0.5·max`. `area` conserva la
  forma pero diluye regiones chicas; `max` conserva el pico pero engorda el área.
  El promedio conserva las dos cosas.
- **Confianza: `area`** (interesa la distribución, no los picos).
- **RGB: bilinear con antialias.**
- Todo se guarda en **uint8**: 1/255 de precisión es de sobra para probabilidades.

### 4.3 Extracción

```bash
python src/extract_maps.py --manifest data/manifest.csv --cache cache \
    --variants raw --map-size 384 --max-side 1280
```

- `--variants`: preprocesamientos **antes** de TruFor (`raw`, `jpeg70`, `jpeg90`,
  `blur1`, `resize1280`, `resize1024+jpeg70`; ver `src/preprocess.py`).
- `--max-side`: límite técnico de memoria para la T4 (TruFor trabaja a resolución
  nativa; las fotos de ~3000×4400 hacen OOM). No es una variante científica.
- Salida: `cache/<variante>__s<S>__ms<max>/<stem>.npz` con `map`, `conf`, `score`,
  `rgb` y `orig_hw`. La clave incluye tamaño y max-side, así que cambiar cualquiera
  de los dos no reutiliza mapas viejos.
- Es **reanudable** y escribe de forma atómica.

### 4.4 Empaquetado

```bash
python src/pack_cache.py --cache cache/raw__s384__ms1280 --out packed/raw__s384
```

Junta los `.npz` en un único `.npy` memmap `(N, 5, S, S)` uint8 — canales
`R, G, B, mapa, confianza` — para no abrir 2400 archivos por época. En Colab se
empaqueta en el **disco local** (`/content/packed`) y se regenera en cada sesión.

---

## 5. Entrenamiento

```bash
python src/train.py --manifest data/manifest.csv --packed packed/raw__s384 \
    --out runs/vit --size 384 --epochs 150 --bs 16
```

- **Split por grupo**, estratificado y reproducible (`runs/splits.json`). Se
  reutiliza en todas las corridas salvo `--resplit`.
- **ViT desde cero**: patch embed `Conv2d(C, dim, 16, 16)`, `depth=8`, `dim=384`,
  `heads=6`, pre-LN, SDPA, dropout + stochastic depth + LayerScale (~15 M params).
- **Prior de TruFor**: el score global entra como token extra y el logit final es
  `vit_logit + alpha·logit(score)` con `alpha=1`. En la época 0 el modelo equivale
  a TruFor; solo aprende una corrección. `--no-score` / `--no-residual` lo apagan
  para la ablación.
- **Canales**: `--channels maps+rgb` (5) o `--channels maps` (2, solo los mapas).
- **Optimización**: AdamW (sin weight decay en normas/biases/tokens), warmup +
  coseno, AMP fp16, grad clipping, EMA opcional, early stopping por AUC.
- **Augment**: solo geométrico (flips + rot90), aplicado igual a los 5 canales.
  Nada fotométrico después de TruFor: desincronizaría los mapas. Para simular
  recompresión hay que usar una **variante** y volver a extraer.

Salidas en `--out`: `last.pt` (cada época, reanudable), `best.pt` (mejor AUC),
`config.json`, `log.csv`.

### 5.1 Presupuesto en T4

| paso | tiempo aprox. |
|---|---|
| descarga IMD2020 (600 MB) | 1–2 min |
| preparación + manifest | 1–2 min |
| extracción TruFor (2424 imgs, max-side 1280) | 30–60 min |
| empaquetado local | ~2 min |
| 150 épocas (S=384, bs=16) | 1–1.5 h |

## 6. Evaluación

```bash
python src/eval.py --manifest data/manifest.csv --packed packed/raw__s384 \
    --ckpt runs/vit/best.pt --out runs/eval --split val
```

Imprime accuracy, balanced accuracy, F1, AUC y AP; guarda `metrics.json`,
`predictions.csv`, `metricas.png` (matriz de confusión + ROC + PR) y `galeria.png`
(errores más confiados primero).

Para un **test externo** (otro dataset, fotos propias) usá `--split all` con su
propio manifest y cache:

```bash
python src/eval.py --manifest /ruta/externo/manifest.csv --packed packed/ext \
    --ckpt runs/vit/best.pt --out runs/eval_ext --split all
```

---

## 7. Flujo completo en Colab (T4)

`notebooks/train_colab.ipynb` tiene el workflow listo. Resumen:

1. Verificar GPU; montar Drive (`BASE=/content/drive/MyDrive/pdi`).
2. Clonar el repo e instalar dependencias.
3. `bash scripts/get_trufor.sh` + smoke test de TruFor.
4. Descargar IMD2020 y prepararlo en `{BASE}/data`.
5. Extraer mapas a `{BASE}/cache` (reanudable).
6. Empaquetar en `/content/packed` (local).
7. Prueba rápida (3 épocas) y entrenamiento completo (reanudable).
8. Evaluar y ver métricas/galería.

Si la sesión se corta: re-ejecutar 1→4, el empaquetado y el entrenamiento; retoma
desde `last.pt`.

---

## 8. Advertencias metodológicas

1. **TruFor se entrenó con IMD2020.** Las métricas sobre ese dataset están
   infladas: el modelo ya vio esas imágenes. El número honesto es el de un test
   externo (CASIA, Columbia, CoMoFoD, fotos propias).
2. **Fuga original/manipulada.** Resuelta con el split por grupo y el
   agrupamiento por contenido (NCC fuera de la máscara); si agregás datos,
   regenerá el split (`--resplit`). Queda un resto inevitable: **148 de las 414
   auténticas** del repack de Kaggle no tienen ninguna manipulada relacionada,
   así que no hay nada que agrupar. Tampoco hay metadatos EXIF (el repack los
   borró), que habrían dado el emparejamiento exacto.
3. **El repack de Kaggle está desordenado.** Nombres tipo identificador sin
   relación con la escena, 45 pares de archivos duplicados literalmente y hasta
   17 versiones de una misma foto bajo ids distintos. Si aparece un release
   mejor, `prepare_imd2020.py` lo acepta igual: el agrupamiento por contenido no
   depende de los nombres.
4. **Desbalance 5:1.** Reportar balanced accuracy, F1, AUC y AP.
5. **ViT desde cero con ~2400 imágenes** es el punto más débil del diseño. El
   prior + logit residual garantizan no arrancar peor que TruFor. Si no alcanza,
   la palanca es pasar a un backbone preentrenado (`timm`) sin cambiar el resto.
6. **TruFor es de uso no comercial.** Hay que citar a los autores si se publican
   resultados (Guillaro et al., *TruFor: Leveraging all-round clues for trustworthy
   image forgery detection and localization*, CVPR 2023).

---

## 9. Estructura del repositorio

```
src/
  detect.py            carga de TruFor y analyze() (único lugar que lo invoca)
  preprocess.py        variantes de preprocesamiento ANTES de TruFor
  maps.py              normalización de tamaño de los mapas DESPUÉS de TruFor
  prepare_imd2020.py   Kaggle -> images/masks + manifest con grupos
  extract_maps.py      pasada de TruFor, cache reanudable
  pack_cache.py        .npz -> memmap .npy
  dataset.py           dataset, split por grupo, augment
  model.py             ViT clasificador desde cero
  train.py             entrenamiento + métricas + checkpoints
  eval.py              evaluación, curvas y galería
scripts/
  get_trufor.sh        código + pesos de TruFor
  make_pruebas.py      imágenes de prueba con ground-truth conocido
notebooks/
  train_colab.ipynb    workflow de entrenamiento en Colab (T4)
```
