import os
import logging
import threading
import uuid
import tempfile
import zipfile
from io import BytesIO

from flask import (
    Flask, request, redirect, url_for, render_template, send_file
)
from werkzeug.utils import secure_filename

import cv2
from cv2 import data
import numpy as np
from keras_facenet import FaceNet

# SQLAlchemy
from sqlalchemy import (
    create_engine, Column, String, Integer, Text, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ─── ЛОГИРОВАНИЕ ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ─── НАСТРОЙКА БАЗЫ ────────────────────────────────────────────────────────────
DATABASE_URL = os.environ['DATABASE_URL']
engine     = create_engine(DATABASE_URL, echo=False)
Session    = sessionmaker(bind=engine)
Base       = declarative_base()

class Task(Base):
    __tablename__ = 'tasks'
    job_id    = Column(String, primary_key=True, index=True)
    status    = Column(String, default='processing', nullable=False)
    processed = Column(Integer, default=0)
    total     = Column(Integer, default=0)
    found     = Column(JSON,   default=[])
    zip_path  = Column(String, nullable=False)
    face_path = Column(String, nullable=False)
    error     = Column(Text,   nullable=True)

Base.metadata.create_all(bind=engine)

# ─── FLASK & GLOBALS ──────────────────────────────────────────────────────────
app     = Flask(__name__)
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ─── ФОН – долгий код обработки ────────────────────────────────────────────────
def long_task(job_id: str):
    session = Session()
    task    = session.query(Task).get(job_id)

    try:
        logging.info(f"[{job_id}] Распаковка {task.zip_path}")
        with zipfile.ZipFile(task.zip_path, 'r') as z:
            work_dir = os.path.dirname(task.zip_path)
            z.extractall(work_dir)
        logging.info(f"[{job_id}] Архив распакован в {work_dir}")

        # файлы для обработки
        files = [
            f for f in os.listdir(work_dir)
            if f.lower().endswith(('.jpg','jpeg','png'))
        ]
        task.total = len(files)
        session.commit()
        logging.info(f"[{job_id}] Всего файлов: {task.total}")

        embedder   = FaceNet()
        sample_img = cv2.imread(task.face_path)
        faces      = cascade.detectMultiScale(
            sample_img, scaleFactor=1.1, minNeighbors=4
        )
        if not len(faces):
            raise RuntimeError("На фото-образце не найдено лицо")
        x,y,w,h        = faces[0]
        sample_face    = cv2.resize(sample_img[y:y+h, x:x+w], (160,160))
        sample_emb     = embedder.embeddings([sample_face])[0]
        logging.info(f"[{job_id}] Эталонный эмбеддинг получен")

        found = []
        for idx, fname in enumerate(files, start=1):
            task.processed = idx
            session.commit()
            logging.info(f"[{job_id}] {idx}/{task.total}: {fname}")

            img = cv2.imread(os.path.join(work_dir, fname))
            if img is None:
                continue
            faces = cascade.detectMultiScale(img, 1.1, 4)
            if not len(faces):
                continue

            x,y,w,h   = faces[0]
            face      = cv2.resize(img[y:y+h, x:x+w], (160,160))
            emb       = embedder.embeddings([face])[0]
            cos_sim   = np.dot(sample_emb, emb) / (
                np.linalg.norm(sample_emb) * np.linalg.norm(emb)
            )
            if cos_sim >= 0.5:
                found.append(fname)
                logging.info(f"[{job_id}] Совпадение: {fname} (sim={cos_sim:.2f})")

        task.found  = found
        task.status = 'done'
        session.commit()
        logging.info(f"[{job_id}] Готово: найдено {len(found)}/{task.total}")

    except Exception as e:
        logging.exception(f"[{job_id}] Ошибка в задаче")
        task.status = 'error'
        task.error  = str(e)
        session.commit()

    finally:
        session.close()

# ─── РОУТЫ ────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        if 'zip_file' not in request.files or 'face_image' not in request.files:
            return "Нужно ZIP и фото.", 400

        # сохраняем во временный каталог
        work_dir = tempfile.mkdtemp()
        zip_f    = request.files['zip_file']
        face_f   = request.files['face_image']
        zip_path = os.path.join(work_dir, secure_filename(zip_f.filename))
        face_path= os.path.join(work_dir, secure_filename(face_f.filename))
        zip_f.save(zip_path)
        face_f.save(face_path)

        # создаём запись в БД
        job_id = str(uuid.uuid4())
        session= Session()
        task   = Task(
            job_id=job_id,
            zip_path=zip_path,
            face_path=face_path
        )
        session.add(task)
        session.commit()
        session.close()

        logging.info(f"[{job_id}] Получены файлы, запускаем фон")
        threading.Thread(target=long_task, args=(job_id,), daemon=True).start()

        return redirect(url_for('status', job_id=job_id))

    return render_template('index.html')


@app.route('/status/<job_id>')
def status(job_id):
    session = Session()
    task    = session.query(Task).get(job_id)
    session.close()
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
    return f"Ошибка: {task.error}", 500


@app.route('/download/<job_id>')
def download(job_id):
    session = Session()
    task    = session.query(Task).get(job_id)
    session.close()
    if not task or task.status != 'done':
        return "Результат не готов", 404

    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for fname in task.found:
            full = os.path.join(os.path.dirname(task.zip_path), fname)
            z.write(full, arcname=fname)
    buf.seek(0)

    return send_file(
        buf,
        as_attachment=True,
        download_name='found_faces.zip',
        mimetype='application/zip'
    )

# ─── ЗАПУСК ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
