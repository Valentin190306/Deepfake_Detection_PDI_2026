# Sistema de detección de alteraciones en imágenes

Procesamiento Digital de Imágenes (11804) - Universidad Nacional de Luján - 2026, 2do cuatrimestre

Integrantes: Thiago Puyelli, Matias Herrneder, Valentín J. Romero Monteagudo
Profesora: Lic. Sandra Serafino

Detección de **manipulación clásica de imágenes** (splicing, copy-move, eliminación de objetos por inpainting) usando **TruFor**, un modelo preentrenado del grupo GRIP-UNINA.

Para cada imagen el sistema devuelve:

- un **score global** de 0 a 1 (probabilidad de que la imagen esté manipulada),
- un **mapa de localización**: dónde estaría la región editada,
- un **mapa de confianza**: cuánta certeza tiene el modelo en cada zona de ese mapa.

No hay entrenamiento: se usa el checkpoint publicado por los autores.

---

## 1. Requisitos

- **Git**, **curl** y **unzip**
- **uv** (gestor de entornos y paquetes de Python): <https://docs.astral.sh/uv/getting-started/installation/>
  - Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh` y abrir una terminal nueva.
  - openSUSE también lo tiene en zypper (`sudo zypper install python313-uv`).
- No hace falta tener Python instalado en una versión específica: `uv` descarga la que le pidamos.
- GPU NVIDIA opcional. Anda en CPU, solo que más lento.

> Usamos **Python 3.12**. Las versiones muy nuevas (3.13/3.14) suelen no tener aún paquetes de PyTorch compatibles.

## 2. Instalación

Todos los comandos se ejecutan **desde la raíz del repositorio**.

```bash
git clone <URL-DEL-REPO>
cd Deepfake_Detection_PDI_2026

uv venv --python 3.12
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2.1 PyTorch (elegir UNA opción según el hardware)

**Sin GPU (solo CPU, versión liviana):**

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Con GPU NVIDIA:** tomar el comando exacto del selector de <https://pytorch.org/get-started/locally/> (Pip, Linux/Windows, CUDA que corresponda) y anteponerle `uv`:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/<cuXXX>
```

Requiere el driver de NVIDIA instalado en el sistema.

> **Placas RTX 50xx (Blackwell)** necesitan `cu128` o superior. Con una build más vieja PyTorch compila pero falla al usar la GPU.

### 2.2 Resto de dependencias

```bash
uv pip install -r requirements.txt
```

### 2.3 Bajar el modelo TruFor

El código de inferencia y los pesos (~270 MB) **no están en el repositorio**: los baja este script a `third_party/`.

```bash
bash scripts/get_trufor.sh
```

Hace dos cosas: un *sparse checkout* de <https://github.com/grip-unina/TruFor> (solo la carpeta `test_docker/`, que es el paquete de inferencia autocontenido — no hace falta Docker) y la descarga de los pesos oficiales desde el sitio de GRIP-UNINA. Es idempotente: si ya está todo, no vuelve a bajar nada.

### 2.4 Verificar

```bash
python -c "import torch, timm; print(torch.__version__, torch.cuda.is_available())"
python src/detect.py third_party/trufor/images/ --viz
```

El segundo comando corre el detector sobre las 4 imágenes de muestra que trae TruFor, de las que conocemos la respuesta correcta. Tiene que dar:

```
0.1940  AUTENTICA   pristine1.jpg
0.2002  AUTENTICA   pristine2.jpg
0.9962  MANIPULADA  tampered1.png
0.9979  MANIPULADA  tampered2.png
```

Si los scores dan parecidos (no idénticos: GPU y CPU difieren en los decimales), la instalación está bien.

> En cada terminal nueva hay que volver a activar el entorno (`source .venv/bin/activate`) desde la raíz del repo.

## 3. Estructura del repositorio

```md
.
├── src/
│   └── detect.py              # detección: score + mapa de manipulación + confianza
├── scripts/
│   ├── get_trufor.sh          # descarga del modelo y los pesos
│   └── make_pruebas.py        # genera imágenes manipuladas con ground-truth conocido
├── third_party/trufor/        # código y pesos de TruFor, NO se sube a Git
├── notebooks/                 # exploración y visualización de resultados
├── pruebas/                   # imágenes de prueba generadas, NO se sube a Git
├── runs/                      # salidas (PNG y JSON), NO se sube a Git
├── requirements.txt
└── README.md
```

## 4. Uso

Desde la raíz del repo, con el entorno activado:

```bash
# Una sola imagen
python src/detect.py foto.png

# Una carpeta (recursiva), guardando las visualizaciones PNG
python src/detect.py mis_fotos/ --viz --out runs/analisis

# Un glob, con umbral propio
python src/detect.py "mis_fotos/*.jpg" --threshold 0.3

# Todo a JSON para procesar después
python src/detect.py mis_fotos/ --json runs/resultados.json

# Forzar CPU (por ejemplo si la GPU se queda sin memoria)
python src/detect.py foto.png --device cpu
```

Opciones: `--viz` guarda un PNG por imagen con 4 paneles (original, mapa de manipulación, overlay, confianza); `--out` indica dónde; `--json` vuelca scores y `area_sospechosa` (porcentaje del área marcada como editada); `--threshold` es el corte para declarar MANIPULADA (default 0.5).

Imágenes grandes pueden no entrar en la memoria de la GPU. El script detecta el *out of memory* y reintenta esa imagen en CPU automáticamente: es más lento, pero no se cae ni redimensiona (redimensionar arruinaría la detección, ver sección 6).

## 5. Cómo probarlo con casos propios

La prueba que vale para el informe es fabricar manipulaciones de las que **nosotros** conocemos el ground-truth, y medir cómo responde el modelo. Este script genera cuatro casos con PIL sobre una misma imagen base:

```bash
python scripts/make_pruebas.py
python src/detect.py pruebas/ --viz --out runs/pruebas
```

| caso | qué es | score | veredicto |
|---|---|---|---|
| `A_copymove.png` | región copiada y pegada en la misma imagen | 0.999 | MANIPULADA ✓ |
| `B_splicing.png` | pedazo de *otra* imagen pegado | 0.998 | MANIPULADA ✓ |
| `C_control.png` | la imagen base sin tocar | 0.200 | AUTENTICA ✓ |
| `D_splicing_jpeg70.jpg` | el caso B re-guardado en JPEG calidad 70 | 0.926 | MANIPULADA ✓ |

Variando el tamaño de la región pegada, el suavizado de los bordes o la calidad de recompresión se puede caracterizar dónde empieza a fallar.

## 6. Limitaciones que hay que tener en cuenta

Son las que condicionan cómo se arma el experimento y cómo se leen los resultados.

1. **Cada recompresión borra huellas.** El caso `D` de la tabla bajó de 0.998 a 0.926 solo por re-guardarse en JPEG 70. El modelo detecta inconsistencias de bajo nivel (ruido del sensor, artefactos de compresión); guardar en JPEG, redimensionar o aplicar filtros las degrada. **Hay que trabajar siempre con los archivos originales**, nunca con imágenes que pasaron por WhatsApp o por un screenshot.

2. **El mapa de localización se equivoca más que el score global, pero avisa.** En `pristine1.jpg` (auténtica) aparece un foco rojo falso, y el mapa de confianza está oscuro justo ahí. **Nunca leer el heatmap sin mirar la confianza al lado.**

3. **El umbral 0.5 es un default, no una verdad.** TruFor fue entrenado sobre CASIA, IMD2020 y tampCOCO; sobre imágenes de otra naturaleza la calibración se corre. Con un set etiquetado propio conviene medir el AUC y elegir el umbral según el costo relativo de falso positivo vs falso negativo.

4. **GPU y CPU no dan resultados idénticos** (diferencias de punto flotante, del orden de 0.02). Si una imagen queda rozando el umbral, no juzgarla por el tercer decimal: mirar el mapa y la confianza.

5. **Detecta edición clásica, no todo lo "falso".** Para deepfakes de caras o imágenes generadas íntegramente por IA (Stable Diffusion, Midjourney) hacen falta otros modelos.

## 7. Licencia y atribución

El código y los pesos de TruFor son de GRIP-UNINA (Università degli Studi di Napoli Federico II) y están bajo una licencia de **uso sin fines de lucro**, que exige citar a los autores al publicar resultados (ver `third_party/trufor/LICENSE.txt`).

> F. Guillaro, D. Cozzolino, A. Sud, N. Dufour, L. Verdoliva.
> *TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization.*
> CVPR 2023.

## 8. Flujo de trabajo en equipo

- El código reutilizable va en `.py`; los notebooks son solo para explorar y mostrar resultados.
- Limpiar las salidas de los notebooks antes de hacer commit (por ejemplo con `nbstripout`), para que el repo no crezca ni genere conflictos ilegibles.
- Anotar cada caso analizado (imagen, score, veredicto, si era correcto) en una planilla compartida.
- No subir imágenes pesadas, el contenido de `third_party/` ni el entorno `.venv/`.
