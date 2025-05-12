import os
import uuid
import threading
import tempfile
import zipfile
from io import BytesIO

from flask import (
    Flask, render_template, request,
    redirect, url_for, send_file
)
from flask_sqlalchemy import SQLAlchemy
import cv2
from cv2 import data
import numpy as np
from keras_facenet import FaceNet

# ─── Конфиг и инициализация ───────────────────────────────────────────────────
app = Flask(__name__)
# DATABASE_URL берётся из ENV (например, postgresql+pg8000://…)
app.config['SQLALCHEMY_DATABASE_URI']    = os.environ['DATABASE_URL']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Детектор и эмбеддер инициализируем один раз
cascade  = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
embedder = FaceNet()

# ─── Модель задачи ────────────────────────────────────────────────────────────
class Task(db.Model):
    job_id    = db.Column(db.String,  primary_key=True)
    status    = db.Column(db.String,  default='processing', nullable=False)
    processed = db.Column(db.Integer, default=0)
    total     = db.Column(db.Integer, default=0)
    found     = db.Column(db.JSON,    default=[])

# Создаём таблицы (1 раз при старте)
with app.app_context():
    db.create_all()

# ─── Фоновая обработка ────────────────────────────────────────────────────────
def long_task(job_id, zip_path, face_path):
    task     = Task.query.get(job_id)
    work_dir = os.path.dirname(zip_path)

    # 1) Распаковать ZIP
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(work_dir)

    # 2) Собрать список изображений
    files = [
        f for f in os.listdir(work_dir)
        if f.lower().endswith(('.jpg','jpeg','png'))
    ]
    task.total = len(files)
    db.session.commit()

    # 3) Эмбеддинг фото-образца
    img = cv2.imread(face_path)
    faces = cascade.detectMultiScale(img, 1.1, 4)
    if not faces:
        task.status = 'error'
        db.session.commit()
        return

    x, y, w, h = faces[0]
    sample_face = cv2.resize(img[y:y+h, x:x+w], (160,160))
    sample_emb  = embedder.embeddings([sample_face])[0]

    # 4) Перебор, сравнение и сбор совпадений
    found = []
    for idx, fname in enumerate(files, start=1):
        task.processed = idx
        db.session.commit()

        img = cv2.imread(os.path.join(work_dir, fname))
        faces = cascade.detectMultiScale(img, 1.1, 4)
        if not len(faces):
            continue

        x, y, w, h = faces[0]
        face = cv2.resize(img[y:y+h, x:x+w], (160,160))
        emb  = embedder.embeddings([face])[0]

        cos_sim = np.dot(sample_emb, emb) / (
            np.linalg.norm(sample_emb) * np.linalg.norm(emb)
        )
        if cos_sim >= 0.5:
            found.append(fname)

    # 5) Сохраняем результат
    task.found  = found
    task.status = 'done'
    db.session.commit()

# ─── Роуты ────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        if 'zip_file' not in request.files or 'face_image' not in request.files:
            return "Нужно ZIP и фото.", 400

        # Сохраняем файлы во временную папку
        work_dir = tempfile.mkdtemp()
        zip_f    = request.files['zip_file']
        face_f   = request.files['face_image']
        zip_path = os.path.join(work_dir, zip_f.filename)
        face_path= os.path.join(work_dir, face_f.filename)
        zip_f.save(zip_path)
        face_f.save(face_path)

        # Регистрируем задачу в БД и запускаем фон
        job_id = str(uuid.uuid4())
        task   = Task(job_id=job_id)
        db.session.add(task)
        db.session.commit()

        threading.Thread(
            target=long_task,
            args=(job_id, zip_path, face_path),
            daemon=True
        ).start()

        return redirect(url_for('status', job_id=job_id))

    return render_template('index.html')


@app.route('/status/<job_id>')
def status(job_id):
    task = Task.query.get(job_id)
    if not task:
        return "Задача не найдена", 404
    if task.status == 'processing':
        return render_template(
            'status.html',
            job_id=job_id,
            processed=task.processed,
            total=task.total
        )
    if task.status == 'done':
        return redirect(url_for('download', job_id=job_id))
    return "Ошибка при обработке", 500


@app.route('/download/<job_id>')
def download(job_id):
    task = Task.query.get(job_id)
    if not task or task.status != 'done':
        return "Результат не готов", 404

    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        work_dir = tempfile.gettempdir()  # либо os.path.dirname(saved zip)
        for fname in task.found:
            full = os.path.join(work_dir, fname)
            z.write(full, arcname=fname)
    buf.seek(0)

    return send_file(
        buf,
        as_attachment=True,
        download_name='found_faces.zip',
        mimetype='application/zip'
    )

# ─── Старт ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
