import os
import io
import zipfile
import tempfile
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import cv2
from cv2 import data
from keras_facenet import FaceNet

# Инициализация
embedder = FaceNet()  # автоматически загрузит и закеширует веса
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
THRESHOLD = 0.5

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def extract_face(image_path, required_size=(160, 160)):
    # Детектируем лицо с помощью Haarcascade
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = img_bgr[y:y+h, x:x+w]
    face = cv2.resize(face, required_size)
    # Конвертируем в RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return face


def cosine_similarity(a, b):
    from numpy import dot
    from numpy.linalg import norm
    return dot(a, b) / (norm(a) * norm(b))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        zip_file = request.files.get('zip_file')
        face_image = request.files.get('face_image')
        if not zip_file or not face_image:
            return 'Оба файла (zip и фото) должны быть загружены.'

        with tempfile.TemporaryDirectory() as tmpdir:
            # Сохранение
            zip_path = os.path.join(tmpdir, secure_filename(zip_file.filename))
            face_path = os.path.join(tmpdir, secure_filename(face_image.filename))
            zip_file.save(zip_path)
            face_image.save(face_path)

            # Распаковка
            extract_dir = os.path.join(tmpdir, 'photos')
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

            # Обработка образца
            sample_face = extract_face(face_path)
            if sample_face is None:
                return 'На фото‑образце не найдено лицо.'
            sample_emb = embedder.embeddings([sample_face])[0]

            # Поиск совпадений
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

            # Формируем zip-ответ
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf_out:
                for m in matches:
                    zf_out.write(m, arcname=os.path.basename(m))
            buf.seek(0)

        return send_file(buf,
                         mimetype='application/zip',
                         as_attachment=True,
                         download_name='found_faces.zip')

    return render_template('index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)