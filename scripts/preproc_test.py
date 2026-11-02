import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import detect

SUAVIZADO = ("suavizado", True, {"metodo": "bilateral", "d": 5, "sigma_color": 25, "sigma_space": 5})
BRILLO    = ("brillo",    True, {"factor": 1.2, "offset": 0, "gamma": 1.0})
BLANCOS   = ("blancos",   True, {"umbral": 200, "fuerza": 0.5})

CONFIGS = {
    "sin_preproc": [],
    "suavizado":   [SUAVIZADO],
    "blancos":     [BLANCOS],
    "brillo":      [BRILLO],
    "las_3":       [SUAVIZADO, BRILLO, BLANCOS],
}

detect.run_experiments(sys.argv[1], CONFIGS, out="runs/preproc_test.csv")
