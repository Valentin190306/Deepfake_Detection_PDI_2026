"""
make_pruebas.py - Genera imágenes manipuladas de prueba con ground-truth conocido.

La idea: fabricar nosotros las manipulaciones para saber con certeza cuál es la
respuesta correcta, y así poder medir al detector en vez de solo mirarlo.

    python scripts/make_pruebas.py
    python src/detect.py pruebas/ --viz --out runs/pruebas

Los cuatro casos están pensados para contrastar entre sí:
  A  copy-move  -> región duplicada dentro de la MISMA imagen
  B  splicing   -> región traída de OTRA imagen
  C  control    -> la base sin tocar (no debe dar manipulada)
  D  B en JPEG  -> mismo splicing, recomprimido: muestra cuánta señal se pierde
"""
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
IMAGES = REPO / "third_party" / "trufor" / "images"
OUT = REPO / "pruebas"


def main():
    base_path = IMAGES / "pristine2.jpg"
    if not base_path.exists():
        raise SystemExit("Faltan las imágenes de TruFor. Correr primero: bash scripts/get_trufor.sh")

    OUT.mkdir(exist_ok=True)
    base = Image.open(base_path).convert("RGB")
    W, H = base.size

    # --- A: copy-move -----------------------------------------------------
    # Duplicar una región dentro de la misma imagen. Es el caso difícil: la
    # región pegada tiene exactamente las mismas estadísticas de ruido y de
    # compresión que el resto, porque viene de la misma foto.
    a = base.copy()
    region = base.crop((int(W * 0.15), int(H * 0.55), int(W * 0.35), int(H * 0.75)))
    a.paste(region, (int(W * 0.55), int(H * 0.55)))
    a.save(OUT / "A_copymove.png")

    # --- B: splicing ------------------------------------------------------
    # Pegar un trozo de otra imagen: distinta cámara, distinto historial de
    # compresión. Deja huellas más marcadas que el copy-move.
    b = base.copy()
    otra = Image.open(IMAGES / "pristine1.jpg").convert("RGB")
    trozo = otra.crop((40, 40, 200, 160)).resize((int(W * 0.25), int(H * 0.2)))
    b.paste(trozo, (int(W * 0.6), int(H * 0.2)))
    b.save(OUT / "B_splicing.png")

    # --- C: control -------------------------------------------------------
    # Sin manipular. Si esta da MANIPULADA, el detector tiene falsos positivos.
    base.save(OUT / "C_control.png")

    # --- D: el mismo splicing, recomprimido -------------------------------
    # Mismo contenido que B. La diferencia de score entre ambos es, exactamente,
    # lo que la recompresión JPEG le borra al detector.
    b.save(OUT / "D_splicing_jpeg70.jpg", quality=70)

    print(f"4 imágenes generadas en {OUT}/ (base: {W}x{H})")
    print("Ahora: python src/detect.py pruebas/ --viz --out runs/pruebas")


if __name__ == "__main__":
    main()
