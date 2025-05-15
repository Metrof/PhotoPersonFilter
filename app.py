import os
import io
import uuid
import zipfile
import tempfile
from threading import Thread

from flask import Flask, request, render_template, url_for, send_file, redirect
import cv2
from cv2 import data
import numpy as np
from keras_facenet import FaceNet

app = Flask(__name__)
# Словарь задач: task_id → {current, total, matches, fname, done}
progress = {}

# Инициализация детектора и эмбеддера
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
embedder = FaceNet()
THRESHOLD = 0.5


def find_compares(task_id, z, sample_emb):
    # Считаем только файлы-изображения для total
    image_names = [n for n in z.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total = len(image_names)
    progress[task_id] = {'current': 0, 'total': total, 'done': False}

    matches = []
    for name in image_names:
        data = z.read(name)
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            progress[task_id]['current'] += 1
            continue
        faces = face_cascade.detectMultiScale(img, 1.1, 4)
        if len(faces) == 0:
            progress[task_id]['current'] += 1
            continue
        found = False
        for (x, y, w, h) in faces:
            face = cv2.resize(img[y:y+h, x:x+w], (160, 160))
            emb = embedder.embeddings([face])[0]
            sim = np.dot(sample_emb, emb) / (np.linalg.norm(sample_emb) * np.linalg.norm(emb))
            if sim >= THRESHOLD:
                found = True
                break
        if found:
            matches.append((name, data))
        progress[task_id]['current'] += 1

    # Создаем ZIP в temp dir
    fname = f"found_{uuid.uuid4().hex}.zip"
    tmp = os.path.join(tempfile.gettempdir(), fname)
    with zipfile.ZipFile(tmp, 'w') as out:
        for name, data in matches:
            out.writestr(name, data)

    # Обновляем прогресс
    progress[task_id].update({
        'matches': len(matches),
        'fname': fname,
        'done': True
    })


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        zip_file = request.files.get('zip_file')
        face_file = request.files.get('face_image')
        if not zip_file or not face_file:
            return "Нужно загрузить и ZIP-архив, и фото-образец.", 400

        # Обработка образца
        face_bytes = face_file.read()
        arr = np.frombuffer(face_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        faces = face_cascade.detectMultiScale(img, 1.1, 4)
        if not len(faces):
            return render_template('result.html', count=0, download_url=None,
                                   error="На фото-образце не найдено лицо.")
        x, y, w, h = faces[0]
        crop = cv2.resize(img[y:y+h, x:x+w], (160, 160))
        sample_emb = embedder.embeddings([crop])[0]

        # ZIP
        z = zipfile.ZipFile(io.BytesIO(zip_file.read()))
        task_id = str(uuid.uuid4())
        Thread(target=find_compares, args=(task_id, z, sample_emb)).start()
        return redirect(url_for('status', task_id=task_id))

    return render_template('index.html')

@app.route('/status/<task_id>')
def status(task_id):
    task = progress.get(task_id)
    if not task:
        return "Неверный task_id", 404

    if not task['done']:
        return render_template('status.html',
                               processed=task['current'],
                               total=task['total'],
                               task_id=task_id,
                               done=False)

    # Когда done=True, сразу перенаправляем на результат
    download_url = url_for('download', fname=task['fname'])
    return render_template('result.html',
                           count=task['matches'],
                           download_url=download_url,
                           error=None)

@app.route('/download/<fname>')
def download(fname):
    tmp = os.path.join(tempfile.gettempdir(), fname)
    if not os.path.exists(tmp):
        return "Файл не найден", 404
    return send_file(tmp, as_attachment=True,
                     download_name='found_faces.zip',
                     mimetype='application/zip')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
