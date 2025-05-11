import os
import io
import zipfile
import tempfile
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image

# --- Face detection & embeddings -----------------------------------------
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet  # автоматически скачает веса при первом запуске

# Инициализация моделей
face_detector = MTCNN()
embedder = FaceNet()  # embedder.embeddings() вернёт 512‑размерные эмбеддинги

THRESHOLD = 0.5  # порог косинусного сходства

# --- Flask ----------------------------------------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------------------------------------------------------------

def extract_face(image_path, required_size=(160, 160)):
    """Извлекает первое лицо с фото и возвращает np.ndarray."""
    image = Image.open(image_path).convert("RGB")
    pixels = np.asarray(image)
    results = face_detector.detect_faces(pixels)
    if not results:
        return None
    x1, y1, w, h = results[0]["box"]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = x1 + w, y1 + h
    face = pixels[y1:y2, x1:x2]
    face = Image.fromarray(face).resize(required_size)
    return np.asarray(face)


def cosine_similarity(a, b):
    from numpy import dot
    from numpy.linalg import norm
    return dot(a, b) / (norm(a) * norm(b))

# --------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        zip_file = request.files["zip_file"]
        face_image = request.files["face_image"]

        # Сохраняем во временную директорию
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, secure_filename(zip_file.filename))
            face_path = os.path.join(tmpdir, secure_filename(face_image.filename))
            zip_file.save(zip_path)
            face_image.save(face_path)

            # Распаковываем фото для перебора
            extract_dir = os.path.join(tmpdir, "photos")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

            # Эмбеддинг образца
            sample_face = extract_face(face_path)
            if sample_face is None:
                return "На фото‑образце не найдено лицо."
            sample_emb = embedder.embeddings([sample_face])[0]

            matches = []
            for root, _, files in os.walk(extract_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    face = extract_face(fpath)
                    if face is None:
                        continue
                    emb = embedder.embeddings([face])[0]
                    if cosine_similarity(sample_emb, emb) >= THRESHOLD:
                        matches.append(fpath)

            # Упаковываем найденные фото
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf_out:
                for m in matches:
                    zf_out.write(m, arcname=os.path.basename(m))
            buf.seek(0)

        return send_file(buf, mimetype="application/zip", as_attachment=True, download_name="found_faces.zip")

    return render_template("index.html")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)