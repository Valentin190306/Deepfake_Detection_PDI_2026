# Sistema de detección de alteraciones en imágenes

Procesamiento Digital de Imágenes (11804) - Universidad Nacional de Luján - 2026, 2do cuatrimestre

Integrantes: Thiago Puyelli, Matias Herrneder, Valentín J. Romero Monteagudo
Profesora: Lic. Sandra Serafino

Clasificador de patches **auténtico / manipulado** basado en fine-tuning de modelos preentrenados (ResNet18 como baseline, Vision Transformer como modelo principal). Primera etapa: **eliminación de objetos (inpainting)** con el dataset IMD2020.

---

## 1. Requisitos

- **Git**
- **uv** (gestor de entornos y paquetes de Python): <https://docs.astral.sh/uv/getting-started/installation/>
  - Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh` y abrir una terminal nueva.
  - openSUSE también lo tiene en zypper (`sudo zypper install python313-uv`).
- No hace falta tener Python instalado en una versión específica: `uv` descarga la que le pidamos.

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

**Con GPU NVIDIA:** tomar el comando exacto del selector de <https://pytorch.org/get-started/locally/> (Pip, Linux/Windows, CUDA que corresponda) y anteponerle `uv`. Ejemplo de formato (la versión de CUDA cambia con el tiempo):

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/<cuXXX>
```

Requiere el driver de NVIDIA instalado en el sistema.

### 2.2 Resto de dependencias

```bash
uv pip install -r requirements.txt
```

### 2.3 Verificar

```bash
python -c "import torch, timm; print(torch.__version__, torch.cuda.is_available())"
```

Debe imprimir la versión de PyTorch. `False` es lo esperado si no hay GPU.

> En cada terminal nueva hay que volver a activar el entorno (`source .venv/bin/activate`) desde la raíz del repo.

## 3. Estructura del repositorio

```md
.
├── src/
│   └── train.py               # entrenamiento, validación y evaluación en test
├── scripts/
│   └── extract_patches.py     # (pendiente) extracción de patches + partición por pares
├── notebooks/                 # exploración y visualización de resultados
├── data/                      # NO se sube a Git
├── runs/                      # checkpoints e historiales, NO se sube a Git
├── requirements.txt
└── README.md
```

## 4. Datos

Los datos y los checkpoints **no están en el repositorio** (ver `.gitignore`).

- **IMD2020**: descargar desde la página del dataset, <https://staff.utia.cas.cz/novozada/db>, y verificar allí qué subconjuntos incluye antes de bajar nada (puede ser muy pesado).
- El script de entrenamiento espera patches ya extraídos con esta estructura:

```md
data/
├── train/{authentic,manipulated}/*.png
├── val/{authentic,manipulated}/*.png
└── test/{authentic,manipulated}/*.png
```

Reglas del dataset (importantes):

1. **Partir por par original/modificada, antes de extraer patches.** Todos los patches de una imagen y de su versión modificada van al mismo split. Si no, las métricas salen infladas.
2. **Guardar los patches sin pérdida (PNG).** No usar JPEG ni `Resize`: destruyen las huellas de la manipulación.
3. **La partición se fija una sola vez** (con semilla) y se guarda la lista de imágenes por split en un CSV versionado, para que todos usemos el mismo.
4. **Antes de entrenar**, comparar entre clases formato, calidad JPEG, resolución y EXIF, para descartar que el modelo aprenda diferencias del dataset en lugar de la manipulación.

## 5. Uso

Desde la raíz del repo, con el entorno activado:

```bash
# 1) Prueba de cordura: debe sobreajustar 32 imágenes (loss ~ 0, AUC ~ 1)
python src/train.py --data data --overfit 32 --epochs 30 --batch-size 8

# 2) Baseline con ResNet18
python src/train.py --data data --model resnet18 --epochs 10 --out runs/resnet18

# 3) Fine-tuning de ViT (patches de 224x224)
python src/train.py --data data --model vit_small_patch16_224 --lr 5e-5 --out runs/vit_s

# 4) Retomar un entrenamiento cortado
python src/train.py --data data --model resnet18 --out runs/resnet18 --resume

# 5) Evaluación FINAL en test con el mejor checkpoint
python src/train.py --data data --model resnet18 --out runs/resnet18 --test
```

Cada corrida guarda en su carpeta `--out`: `last.pt` (para retomar), `best.pt` (mejor AUC en validación) e `history.csv` (métricas por época).

**El set de test se usa una sola vez**, cuando ya están cerradas todas las decisiones de modelo e hiperparámetros. Si se ajusta el modelo después de ver el resultado en test, deja de ser una medida honesta.

## 6. Entrenar en Google Colab (GPU gratuita)

Colab ya trae PyTorch instalado. Las sesiones son efímeras: guardar los checkpoints en Drive y copiar el dataset al disco local de Colab (leer miles de archivos chicos desde Drive es muy lento).

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone <URL-DEL-REPO>
%cd Deepfake_Detection_PDI_2026
!pip install -q timm scikit-learn pandas

# Dataset empaquetado en un .tar en Drive -> disco local
!mkdir -p /content/data
!tar -xf /content/drive/MyDrive/pdi/patches.tar -C /content/data

# Checkpoints directo a Drive; --resume permite continuar tras una desconexión
!python src/train.py --data /content/data --model resnet18 \
    --out /content/drive/MyDrive/pdi/runs/resnet18 --resume
```

Activar la GPU en *Entorno de ejecución > Cambiar tipo de entorno de ejecución*.

## 7. Flujo de trabajo en equipo

- El código reutilizable va en `.py`; los notebooks son solo para explorar y mostrar resultados.
- Limpiar las salidas de los notebooks antes de hacer commit (por ejemplo con `nbstripout`), para que el repo no crezca ni genere conflictos ilegibles.
- Anotar cada experimento (modelo, hiperparámetros, partición usada, resultados) en una planilla compartida.
- No subir datos, checkpoints (`*.pt`) ni el entorno `.venv/`.
