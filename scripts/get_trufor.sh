#!/usr/bin/env bash
# Baja el modelo TruFor (código de inferencia + pesos preentrenados) a third_party/.
#
# No se versiona nada de esto: son ~270 MB de terceros (ver .gitignore).
# Es idempotente: si ya está bajado, no vuelve a bajarlo.
#
# Uso:  bash scripts/get_trufor.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/third_party/trufor"
WEIGHTS_URL="https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip"

cd "$REPO_ROOT"
mkdir -p third_party

# ---- 1. Código de inferencia -------------------------------------------
# Sparse checkout: del repo solo necesitamos test_docker/, que es el paquete
# de inferencia autocontenido (no hace falta Docker, solo las dependencias
# de requirements.txt).
if [ -d "$DEST/src" ]; then
    echo "[1/2] Código ya presente en third_party/trufor/, salteo."
else
    echo "[1/2] Clonando el código de TruFor..."
    rm -rf third_party/_tmp
    git clone -q --depth 1 --filter=blob:none --sparse \
        https://github.com/grip-unina/TruFor.git third_party/_tmp
    git -C third_party/_tmp sparse-checkout set test_docker
    mv third_party/_tmp/test_docker "$DEST"
    rm -rf third_party/_tmp
fi

# ---- 2. Pesos preentrenados --------------------------------------------
# ~250 MB desde el sitio oficial de GRIP-UNINA (Universidad de Nápoles).
if find "$DEST" -name 'trufor.pth.tar' -print -quit | grep -q .; then
    echo "[2/2] Pesos ya presentes, salteo."
else
    echo "[2/2] Bajando pesos (~250 MB)..."
    tmp_zip="$(mktemp -t trufor_weights_XXXXXX.zip)"
    curl -L --fail -o "$tmp_zip" "$WEIGHTS_URL"
    unzip -o -q "$tmp_zip" -d "$DEST"
    rm -f "$tmp_zip"
fi

echo
echo "Listo. Verificar con:"
echo "  python src/detect.py third_party/trufor/images/ --viz"
echo "Esperado: pristine* ~0.19  |  tampered* ~0.99"
