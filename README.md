# Sistema de detección de alteraciones en imágenes

Procesamiento Digital de Imágenes (11804) - Universidad Nacional de Luján - 2026, 2do cuatrimestre

Integrantes: Thiago Puyelli, Matias Herrneder, Valentín J. Romero Monteagudo
Profesora: Lic. Sandra Serafino

Detección de **manipulación clásica de imágenes** (splicing, copy-move, eliminación
de objetos por inpainting). El sistema combina dos modelos:

1. **TruFor** (Guillaro et al., CVPR 2023, GRIP-UNINA), preentrenado y **congelado**.
   Por cada imagen devuelve un *score* global, un **mapa de manipulación** y un
   **mapa de confianza**.
2. Un **clasificador con backbone preentrenado** (Swin-T de `timm`, con el stem
   inflado a 5 canales) que recibe la imagen junto con los dos mapas de TruFor y
   devuelve la probabilidad de que la imagen esté manipulada. Se compara
   transfer learning (`--ft linear`) contra fine-tuning (`--ft partial/full`), y
   el ViT desde cero de la primera versión queda como control.

El entrenamiento se hace sobre **IMD2020** (Kaggle: `saikotahmed/imd2020`) más
**COCOInpaint** como segundo conjunto, y está pensado para correr en una **GPU T4
de Google Colab**.

```
imagen (tamaño variable)
   │  src/preprocess.py      variantes ANTES de TruFor (raw, jpeg70, blur1, ...)
   ▼
TruFor (congelado)  ──►  score + mapa + confianza  (resolución nativa)
   │  src/maps.py           NORMALIZACIÓN DE TAMAÑO: S×S + uint8
   ▼
cache  ──►  src/pack_cache.py  ──►  memmap local (N, 5, S, S) por dataset
   │  src/merge_manifests.py  une los manifests y marca la base de cada fila
   ▼
Swin-T preentrenado, stem 3→5 (RGB + mapa + confianza)  ──►  P(manipulada)
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

`timm` queda fijo en **1.0.30** porque lo usan las dos partes:

- **TruFor** importa `timm.models.layers` (`DropPath`, `to_2tuple`,
  `trunc_normal_`). En 1.x ese módulo sigue existiendo como alias deprecado de
  `timm.layers`, así que la carga del modelo funciona igual (avisa un
  `FutureWarning`, inofensivo).
- **El clasificador** crea los backbones preentrenados (`swin_t`, `vit_s16`, …).
  Los pesos viven en HuggingFace y timm 0.9.16 no los puede bajar con
  `huggingface_hub` 2.x (falla con `ImportError:
  repo_type_and_id_from_hf_id`), que es lo que trae Colab hoy.

`datasets` es para bajar COCOInpaint (§3.4); `kagglehub`, para IMD2020.

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
> no solo accuracy. Con `--pairs balanced` el conjunto queda 1:1 (ver 3.3).

### 3.3 Escenas con las dos clases (`--pairs`)

El agrupamiento por contenido resuelve la **fuga** (la misma foto en train y en
validación), pero no la **confusión escena ↔ clase**. Si una manipulada sale de
una escena que nunca aparece como auténtica, el modelo puede aprender "esta foto
es trucada" mirando el fondo, el encuadre o el tamaño, en vez de "esta zona está
editada". En IMD2020 eso es la mayoría del dataset: de **1606 escenas, sólo 264
tienen las dos clases**.

`--pairs` recorta el manifest a esas escenas:

| modo | qué deja | filas | imágenes únicas | `pos_weight` |
|---|---|---|---|---|
| `off` (default) | todo | 2424 | 2424 | 0.21 |
| `unique` | sólo escenas mixtas, una fila por archivo | 1042 | 1042 | 0.34 |
| `balanced` | escenas mixtas, **una auténtica por cada variante** | 1552 | 1042 | **1.00** |

```bash
python src/prepare_imd2020.py --src <carpeta_kaggle> --out data --prefix imd_ \
    --group-by content --pairs balanced
```

Detalles de `balanced`:

- Cada escena pesa lo mismo en las dos clases: una escena con 1 auténtica y 3
  variantes aporta 3 auténticas y 3 manipuladas. Es el punto del modo.
- Las copias **apuntan al mismo archivo**, no se duplican bytes. `extract_maps.py`
  deduplica por stem antes de correr TruFor (una inferencia por foto, no 2.9), y
  `train.py` / `eval.py` deduplican la **validación** para que una foto repetida
  no pese varias veces en el umbral ni en las métricas. El train conserva las
  repeticiones a propósito.
- Las 2 escenas con 2 auténticas distintas usan round-robin, así entran las dos.
- Si se descarta una manipulada por tamaño de máscara, su auténtica duplicada
  también se cae, para no romper el 1:1.

**Qué se pierde.** Se descartan 1194 escenas que sólo tienen manipuladas (1234
imágenes) y 148 que sólo tienen auténticas: **−61% de las manipuladas**. Con
`--val-frac 0.25` la validación queda con **61 auténticas únicas** y 61 escenas
(400 filas antes de deduplicar); con 61 negativos el AUC se mueve ±0.05 con
cualquier cambio, y `train.py` elige el mejor checkpoint por AUC, así que puede
quedarse con una época de suerte. Si el número final importa, repetir con 3
semillas (`--seed`) y reportar media ± desvío. Aun así, con 264 escenas de IMD el
techo de generalización es bajo: eso es exactamente lo que motiva sumar COCOInpaint
(§3.4) en vez de confiar sólo en este recorte.

> **Al cambiar de modo hay que borrar el split viejo** (`runs/splits.json`). Los
> grupos pasan de 1606 a 264 y `apply_splits` falla con un mensaje explícito
> ("El split no conoce N grupos") en vez de arrancar en silencio.

### 3.4 COCOInpaint (segundo dataset)

IMD2020 solo no alcanza: es el set de entrenamiento de TruFor, así que cualquier
métrica ahí está inflada. El segundo dataset es
[`phiyodr/InpaintCOCO`](https://huggingface.co/datasets/phiyodr/InpaintCOCO):

| campo | qué es |
|---|---|
| `coco_image` | la foto **original** de COCO (512×512) → auténtica |
| `inpaint_image` | la misma foto con un objeto borrado y rellenado → manipulada |
| `mask` | la zona rellenada |
| `coco_details.coco_url` | de qué foto de COCO salió (sirve de id de escena) |

Son **1260 pares** (split `test`, ~1 GB) con la clase auténtica incluida: no hay
que bajar COCO aparte. La manipulación la generó un modelo de inpainting, no una
persona; es otra familia distinta de IMD2020, y por eso sirve como test honesto
(TruFor no la vio).

```bash
python src/prepare_coco_inpaint.py --out data/coco_inpaint
python src/extract_maps.py --manifest data/coco_inpaint/manifest.csv \
    --cache cache/coco --variants raw --map-size 384
python src/pack_cache.py --cache cache/coco/raw__s384 --out packed/coco__s384
```

Detalles:

- Se guarda en **PNG** (sin pérdida): re-comprimir a JPEG metería artefactos que
  no estaban en la imagen.
- La auténtica se escribe una sola vez por escena, pero se emite **una fila por
  variante manipulada** (igual que `--pairs balanced`): el conjunto queda 1:1 y
  `pos_weight ≈ 1`.
- `group` es el nombre de la foto de COCO, así que dos inpainting de la misma
  escena no se separan entre train y validación.

### 3.5 Mezclar datasets (`merge_manifests.py`)

Los dos caches viven en arrays distintos y pueden compartir nombres de archivo
(COCO numera sus fotos). En vez de fusionar los `.npy`, cada fila del manifest
lleva su base en la columna `packed` y `MapDataset` resuelve a qué array
pertenece.

```bash
python src/merge_manifests.py --out data/manifest_all.csv \
    --input imd  data/manifest.csv              packed/imd__s384 \
    --input coco data/coco_inpaint/manifest.csv packed/coco__s384
```

El script agrega `dataset`, prefija `group` con el nombre del conjunto
(`imd:...` / `coco:...`) y convierte `file`/`mask` a rutas absolutas (los
manifests originales son relativos a su propia carpeta). El resultado es
específico de la máquina, pero se regenera en segundos.

Antes de escribir nada valida que existan los `.npy` y `.json` de cada
`--input`: si falta uno, falla en el merge nombrando el dataset, en vez de dejar
un manifest que explota recién en `train.py`.

Con la mezcla natural (sin submuestrear) quedan ~3100 escenas y ~4072 filas,
60% COCO / 40% IMD. El split estratifica por `(dataset, etiqueta)`, así que
ningún conjunto queda con una sola clase en validación.

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
# un cache por dataset (las claves de cache no se mezclan)
python src/extract_maps.py --manifest data/manifest.csv \
    --cache cache --variants raw --map-size 384 --max-side 1280
python src/extract_maps.py --manifest data/coco_inpaint/manifest.csv \
    --cache cache/coco --variants raw --map-size 384
```

- `--variants`: preprocesamientos **antes** de TruFor (`raw`, `jpeg70`, `jpeg90`,
  `blur1`, `resize1280`, `resize1024+jpeg70`; ver `src/preprocess.py`).
- `--max-side`: límite técnico de memoria para la T4 (TruFor trabaja a resolución
  nativa; las fotos de ~3000×4400 hacen OOM). No es una variante científica. Las
  de COCOInpaint ya son 512×512, así que se puede omitir.
- Salida: `cache/<variante>__s<S>__ms<max>/<stem>.npz` con `map`, `conf`, `score`,
  `rgb` y `orig_hw`. La clave incluye tamaño y max-side, así que cambiar cualquiera
  de los dos no reutiliza mapas viejos.
- Es **reanudable** y escribe de forma atómica. Deduplica por `stem`, así que un
  manifest que repite la auténtica una vez por variante no procesa la misma foto
  varias veces.

### 4.4 Empaquetado

```bash
# uno por dataset: el .npy no se puede fusionar sin renombrar archivos
python src/pack_cache.py --cache cache/raw__s384__ms1280 --out packed/imd__s384
python src/pack_cache.py --cache cache/coco/raw__s384  --out packed/coco__s384
```

Junta los `.npz` en un único `.npy` memmap `(N, 5, S, S)` uint8 — canales
`R, G, B, mapa, confianza` — para no abrir 2400 archivos por época. En Colab se
empaqueta en el **disco local** (`/content/packed`) y se regenera en cada sesión.

Cada dataset va a su propio paquete; `merge_manifests.py` (§3.5) deja en el
manifest la columna `packed` que dice cuál usar por fila. `pack_cache.py` empaqueta
**todos** los `.npz` del directorio, así que el `--cache` tiene que ser el de una
sola clave (un dataset), no la carpeta padre.

---

## 5. Entrenamiento

Dos arquitecturas, elegidas con `--backbone`:

- **`--backbone swin_t` (default)**: backbone preentrenado de timm, con el stem
  inflado a 5 canales. Es el camino nuevo.
- **`--backbone scratch`**: el ViT chico desde cero de la primera versión. Queda
  como control: con ~1000 imágenes únicas colapsó a calibrar el score de TruFor
  (`alpha≈1`, spearman −0.96, AUC = baseline).

| `--backbone` | modelo de timm | params | por qué |
|---|---|---|---|
| `swin_t` | `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k` | 28M | jerárquico y local: ventaja con pocos datos |
| `vit_s16` | `vit_small_patch16_384.augreg_in21k_ft_in1k` | 22M | control de arquitectura (atención global) |
| `vit_b16` | `vit_base_patch16_384.augreg_in21k_ft_in1k` | 86M | si el dataset crece mucho |
| `convnext_t` | `convnext_tiny.in12k_ft_in1k` | 28M | control CNN |

### 5.1 Transfer learning vs fine-tuning (`--ft`)

| `--ft` | qué se congela | qué se entrena | qué mide |
|---|---|---|---|
| `linear` | todo el backbone | stem, norma final, cabeza | cuánto valen los features preentrenados |
| `partial` | primeras etapas/bloques | + las últimas (1 etapa, o 4 bloques en un ViT) | el punto medio |
| `full` | nada | todo, con `--lr-backbone` y decaimiento por capa | fine-tuning real |

`--unfreeze-blocks N` cambia cuántas etapas/bloques se descongelan en `partial`.

El **stem siempre queda entrenable**, incluso en `linear`: los dos canales
nuevos (mapa y confianza) arrancan en cero y, si se congela el stem, los mapas
no pueden entrar al modelo. Por eso `linear` acá no es un linear probe puro: es
"features congelados + una puerta de entrada nueva".

En `linear` el backbone además se queda en modo `eval`: con los pesos congelados,
`train()` sólo activaría su dropout y su stochastic depth, o sea features que
cambian al azar en cada época y una cabeza aprendiendo sobre ruido. En `partial`
y `full` sí se activa, porque las capas descongeladas se regularizan normalmente.

```bash
# E1: transfer learning puro (sólo se entrena la cabeza)
python src/train.py --manifest data/manifest_all.csv --backbone swin_t \
    --ft linear --lr 1e-3 --out runs/swin_linear

# E2: partial (el que suele ganar con datasets chicos)
python src/train.py ... --backbone swin_t --ft partial --out runs/swin_partial

# E3: fine-tuning completo
python src/train.py ... --backbone swin_t --ft full --out runs/swin_full

# control: el ViT desde cero
python src/train.py ... --backbone scratch --drop 0.15 --out runs/scratch
```

### 5.2 El resto de las piezas

- **Split por grupo**, estratificado por `(dataset, etiqueta)` y reproducible
  (`runs/splits.json`). Se reutiliza en todas las corridas salvo `--resplit`.
  `--val-frac` (default 0.25) sólo se usa al crear el split; si el archivo ya
  existe, manda el guardado.
- **Manifest pareado**: si viene de `--pairs balanced`, el train conserva las
  auténticas repetidas (es el punto) y la validación se deduplica por foto.
  `pos_weight` sale ~1.0, así que la loss no compensa nada.
- **Cinco canales**: el stem del backbone se infla 3→5 copiando los pesos RGB y
  dejando los canales nuevos en cero. En la época 0 el modelo es exactamente el
  preentrenado; después los canales nuevos aprenden (el gradiente es `x·δ` y el
  mapa no es cero).
- **Prior de TruFor**: el score global entra como token extra (o sumado al
  vector de features) y el logit final es `vit_logit + alpha·logit(score)` con
  `alpha=1`. En la época 0 el modelo equivale a TruFor; solo aprende una
  corrección. `--no-score` / `--no-residual` lo apagan para la ablación.
- **Loss auxiliar** (`--aux-weight`, default 0.5): supervisa también `vit_logit`
  (la rama sin el prior). Sin ella el residual tiene una salida trivial: dejar la
  rama en cero y sólo atenuar el score de TruFor. Medido en el primer run, el ViT
  terminó con **Spearman −0.96** contra el prior y **AUC idéntico al baseline**:
  aprendió a calibrar, no a detectar. Con `--aux-weight 0` se reproduce ese
  comportamiento; con `--no-residual` no aplica.
- **Canales**: `--channels maps+rgb` (5) o `--channels maps` (2, sólo los mapas).
  Con 2 canales el stem se inicializa promediando los kernels RGB, así que el
  transfer es más débil pero no nulo; sirve como ablación limpia cuando el miedo
  es que el modelo detecte el generador en la imagen en vez de la edición.
- **Optimización**: AdamW con tres velocidades (cabeza y stem con `--lr`,
  backbone con `--lr-backbone`, default `lr/10`, y decaimiento por capa
  `--layer-decay`), sin weight decay en normas/biases/tokens, warmup + coseno,
  AMP fp16, grad clipping, EMA opcional, early stopping por AUC.
- **Augment**: solo geométrico (flips + rot90), aplicado igual a los 5 canales.
  Nada fotométrico después de TruFor: desincronizaría los mapas. Para simular
  recompresión hay que usar una **variante** y volver a extraer.

Salidas en `--out`: `last.pt` (cada época, reanudable), `best.pt` (mejor AUC),
`config.json`, `log.csv`.

### 5.3 Presupuesto en T4

| paso | tiempo aprox. |
|---|---|
| descarga IMD2020 (600 MB) | 1–2 min |
| preparación + manifest | 1–2 min |
| extracción TruFor (2424 imgs, max-side 1280) | 30–60 min |
| extracción TruFor con `--pairs balanced` (1042 imgs) | 25–30 min |
| descarga + preparación COCOInpaint (1260 pares, 512×512) | 3–5 min |
| extracción TruFor COCOInpaint (~2520 imgs, 512×512) | 10–15 min |
| empaquetado local (los dos datasets) | ~3 min |
| 150 épocas Swin-T `partial` (S=384, bs=16, ~4000 filas) | 1.5–2.5 h |
| 150 épocas Swin-T `linear` (sólo cabeza) | 40–60 min |

`linear` es la corrida más barata y la que primero hay que hacer: si los features
congelados ya separan, el fine-tuning tiene poco margen y el problema no es la
arquitectura.

## 6. Evaluación

```bash
python src/eval.py --manifest data/manifest_all.csv \
    --ckpt runs/swin_partial/best.pt --out runs/eval --split val --bootstrap 500
```

Imprime accuracy, balanced accuracy, F1, AUC y AP; guarda `metrics.json`,
`predictions.csv` (con la columna `dataset`), `metricas.png` (matriz de confusión
+ ROC + PR) y `galeria.png` (errores más confiados primero).

**El umbral importa más que el modelo.** La validación pareada es 77% manipulada
(200 positivas y 61 auténticas), así que:

- `best_threshold()` elige el umbral por **balanced accuracy**, no por F1. Con F1,
  la prevalencia corre el óptimo hacia predecir positivo y marca como trucada casi
  cualquier foto buena (medido: thr 0.15 → 85% de falsos positivos sobre las
  auténticas). El checkpoint guarda `thr` (balanceado) y `thr_f1` (para comparar).
- `eval.py` imprime la **matriz de confusión** y el **TPR a FPR fijo** (1%, 5%, 10%),
  que no dependen del umbral elegido. En uso real, "cuántas trucadas detecto si
  acepto 5% de falsos positivos" es la pregunta, no un umbral único.
- `--threshold 0.6` fuerza un umbral; `--thr-f1` usa el de F1 guardado.

`--bootstrap N` agrega un IC 95% de AUC y balanced accuracy remuestreando
**escenas**, no imágenes: las ~3 variantes de una misma foto no son
independientes, así que un bootstrap por fila da intervalos demasiado angostos.
Con 61 escenas en validación el IC es ancho (±0.05 o más) y conviene reportarlo.

### 6.1 Métricas por dataset y test cross-dataset

Cuando el manifest mezcla conjuntos, `eval.py` separa las métricas solo
(`--by-dataset`, o automático si el manifest tiene más de un valor en `dataset`):

```
por dataset:
  coco     n= 812 pos= 406 | bal_acc 0.87 | AUC 0.91 | AP 0.90 | ...
  imd      n= 271 pos= 136 | bal_acc 0.95 | AUC 0.97 | AP 0.96 | ...
```

Esta es la lectura que importa: **IMD2020 es el set de entrenamiento de TruFor**,
así que su número está inflado; **COCO val es out-of-domain** y es el que dice si
el clasificador generaliza.

`--dataset` filtra filas, y combinado con `--split all` sirve de test
cross-dataset (entrenar con uno, medir con el otro):

```bash
# entrenado con IMD+COCO, medido sólo en COCO (ignora el split guardado)
python src/eval.py --manifest data/manifest_all.csv \
    --ckpt runs/swin_partial/best.pt --out runs/eval_coco --split all --dataset coco

# test externo de verdad: otro manifest y su propio cache
python src/eval.py --manifest /ruta/externo/manifest.csv --packed packed/ext \
    --ckpt runs/swin_partial/best.pt --out runs/eval_ext --split all
```

Si el nombre no existe en el manifest, el script falla listando los valores que
sí hay, en vez de evaluar sobre cero filas.

---

## 7. Flujo completo en Colab (T4)

`notebooks/train_colab.ipynb` tiene el workflow listo. Resumen:

1. Verificar GPU; montar Drive (`BASE=/content/drive/MyDrive/pdi`).
2. Clonar el repo e instalar dependencias.
3. `bash scripts/get_trufor.sh` + smoke test de TruFor.
4. Descargar IMD2020 y prepararlo en `{BASE}/data` (`--pairs balanced`).
5. Bajar y preparar COCOInpaint en `{BASE}/data/coco_inpaint`.
6. Extraer mapas de los dos datasets a `{BASE}/cache` (reanudable).
7. Empaquetar cada uno en `/content/packed` (local) y unir los manifests con
   `merge_manifests.py`.
8. Prueba rápida (`--ft linear --limit-steps 20`) y después la corrida real.
9. Evaluar con `--by-dataset` y ver métricas/galería.

Si la sesión se corta: re-ejecutar 1→4, el empaquetado y el entrenamiento; retoma
desde `last.pt`.

La variable `RUN` del notebook elige la corrida (`swin_linear`, `swin_partial`,
`swin_full`, `scratch`) y se usa en las celdas de entrenamiento y evaluación, así
que no hay que editar rutas a mano para comparar.

---

## 8. Advertencias metodológicas

1. **TruFor se entrenó con IMD2020.** Las métricas sobre ese dataset están
   infladas: el modelo ya vio esas imágenes. El número honesto es el de un test
   externo (COCOInpaint, CASIA, Columbia, CoMoFoD, fotos propias). Por eso el
   flujo mezcla datasets y `eval.py --by-dataset` reporta COCO por separado.
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
4. **Desbalance 5:1** en el manifest completo de IMD2020. Con `--pairs balanced`
   queda 1:1 (776/776), a costa de usar 264 escenas y 1042 imágenes únicas. Con
   la mezcla natural de los dos datasets el conjunto queda ~60% COCO / 40% IMD y
   ~1:1 por escena. En todos los casos reportar balanced accuracy, F1, AUC y AP.
5. **COCOInpaint mide "pasó por un generador", no sólo "esta zona se rellenó".**
   Las manipuladas las produjo un modelo de inpainting, mientras que las
   auténticas son fotos de COCO sin pasar por él. Un clasificador puede aprender
   la firma del generador en vez de la edición: si el AUC en COCO es muy alto y en
   IMD2020 no, sospechá de eso. La ablación limpia es
   `--channels maps --no-score --no-residual`: ahí el modelo sólo ve los mapas de
   TruFor, no la imagen que delata al generador.
6. **`--ft linear` no es un linear probe puro.** El stem queda entrenable en todos
   los modos (si se congela, los mapas no entran al modelo). Entrena ~11k params
   en Swin-T (stem 7.7k + norma 1.5k + cabeza), así que mide "features congelados
   + una puerta de entrada nueva". Para un linear probe de verdad habría que
   descartar los mapas y usar `--channels` RGB puro, que no está implementado.
7. **ViT desde cero con ~1000–2400 imágenes** fue el punto más débil del primer
   diseño. El prior + logit residual garantizaban no arrancar peor que TruFor,
   pero también ofrecían una salida trivial: copiar el score. En el primer run el
   ViT quedó con **Spearman −0.96** contra el prior y AUC idéntico al baseline;
   la loss auxiliar (`--aux-weight`) es el remedio, y la ablación
   `--no-score --no-residual` dice si los mapas alcanzan por sí solos. Por eso el
   default ahora es un backbone preentrenado y `--backbone scratch` quedó como
   control.
8. **El backbone preentrenado impone su resolución.** Swin-T es `224` y ViT-S es
   `384`; pasarle `img_size` distinto reescala la interpolación posicional, lo que
   funciona pero degrada los features. Por eso `--size` tiene que coincidir con el
   cache empaquetado y con el tamaño con el que se creó el backbone.
9. **TruFor es de uso no comercial.** Hay que citar a los autores si se publican
   resultados (Guillaro et al., *TruFor: Leveraging all-round clues for trustworthy
   image forgery detection and localization*, CVPR 2023).

---

## 9. Estructura del repositorio

```
src/
  detect.py              carga de TruFor y analyze() (único lugar que lo invoca)
  preprocess.py          variantes de preprocesamiento ANTES de TruFor
  maps.py                normalización de tamaño de los mapas DESPUÉS de TruFor
  prepare_imd2020.py     Kaggle -> images/masks + manifest con grupos
  prepare_coco_inpaint.py  HuggingFace InpaintCOCO -> images/masks + manifest
  extract_maps.py        pasada de TruFor, cache reanudable
  pack_cache.py          .npz -> memmap .npy
  merge_manifests.py     une manifests de varios datasets (columna packed)
  dataset.py             dataset multi-packed, split por grupo, augment
  backbone.py            registry timm, inflado del stem 3->5, modos de congelado
  model.py               clasificador (backbone preentrenado o ViT desde cero)
  train.py               entrenamiento + métricas + checkpoints
  eval.py                evaluación, curvas, métricas por dataset y galería
scripts/
  get_trufor.sh          código + pesos de TruFor
  make_pruebas.py        imágenes de prueba con ground-truth conocido
notebooks/
  train_colab.ipynb      workflow de entrenamiento en Colab (T4)
```
