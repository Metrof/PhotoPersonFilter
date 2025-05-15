import os
import io
import uuid
import zipfile
import tempfile

from flask import Flask, request, render_template, url_for, send_file
import cv2
from cv2 import data
import numpy as np
from keras_facenet import FaceNet

app = Flask(__name__)

# ─── Инициализация face-detector и эмбеддера ───────────────────────────────
embedder     = FaceNet()  # скачивает и кеширует веса при первом запуске
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
THRESHOLD = 0.5

# ─── Главная страница: загрузка ZIP и фото ───────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        zip_file   = request.files.get('zip_file')
        face_file  = request.files.get('face_image')
        if not zip_file or not face_file:
            return "Нужно загрузить и ZIP-архив, и фото-образец.", 400

        # 1) Обрабатываем образец
        face_bytes = face_file.read()
        arr = np.frombuffer(face_bytes, np.uint8)
        sample_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        faces = face_cascade.detectMultiScale(sample_img, 1.1, 4)
        if len(faces) == 0:
            return render_template('result.html', count=0, error="На фото-образце не найдено лицо.")
        x,y,w,h = faces[0]
        crop = cv2.resize(sample_img[y:y+h, x:x+w], (160,160))
        sample_emb = embedder.embeddings([crop])[0]

        # 2) Читаем ZIP и ищем совпадения
        z = zipfile.ZipFile(io.BytesIO(zip_file.read()))
        matches = []
        for name in z.namelist():
            if not name.lower().endswith(('.jpg','.jpeg','.png')):
                continue
            z_data = z.read(name)
            arr = np.frombuffer(z_data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None: 
                continue
            faces = face_cascade.detectMultiScale(img, 1.1, 4)
            if not len(faces):
                continue

            found_this_image = False
            for (x, y, w, h) in faces:
                # вырезаем и приводим к нужному размеру
                face = cv2.resize(img[y:y + h, x:x + w], (160, 160))
                emb = embedder.embeddings([face])[0]
                sim = np.dot(sample_emb, emb) / (np.linalg.norm(sample_emb) * np.linalg.norm(emb))
                if sim >= THRESHOLD:
                    found_this_image = True
                    break  # не обязательно проверять дальше

            if found_this_image:
                matches.append((name, z_data))

        # 3) Запишем найденные фото в временный ZIP и вернём ссылку
        zip_name = f"found_{uuid.uuid4().hex}.zip"
        tmp_path = os.path.join(tempfile.gettempdir(), zip_name)
        with zipfile.ZipFile(tmp_path, 'w') as out:
            for name, z_data in matches:
                out.writestr(name, z_data)

        download_url = url_for('download', fname=zip_name)
        return render_template('result.html',
                               count=len(matches),
                               download_url=download_url,
                               error=None)

    return render_template('index.html')


# ─── Отдача ZIP с найденными фото ────────────────────────────────────────────
@app.route('/download/<fname>')
def download(fname):
    tmp_path = os.path.join(tempfile.gettempdir(), fname)
    if not os.path.exists(tmp_path):
        return "Файл не найден", 404
    return send_file(tmp_path,
                     as_attachment=True,
                     download_name='found_faces.zip',
                     mimetype='application/zip')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
