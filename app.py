import os
import zipfile
import threading
import uuid
from flask import Flask, render_template, request, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from keras_facenet import FaceNet
import cv2
import numpy as np
from cv2 import data

import logging
# Настраиваем логгер
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Инициализация моделей ------------------------------------------------
embedder = FaceNet()  # Автоматически скачивает и кеширует веса
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
THRESHOLD = 0.5

# --- Настройка Flask и папок ----------------------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

tasks = {}  # Хранилище статуса задач

# --- Вспомогательные функции ------------------------------------------------
def extract_face(image_path, required_size=(160, 160)):
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
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return face


def cosine_similarity(a, b):
    from numpy import dot
    from numpy.linalg import norm
    return dot(a, b) / (norm(a) * norm(b))


def process_task(job_id, zip_path, face_path):
    # Инициализируем статус задачи
    logging.info(f"[{job_id}] Задача запущена: распаковка {zip_path}")
    try:
        # 1) Распаковать
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(zip_path))
        logging.info(f"[{job_id}] Архив распакован")

        # 2) Подготовить список файлов
        img_dir = os.path.dirname(zip_path)
        files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
        total = len(files)
        tasks[job_id]['total'] = total
        logging.info(f"[{job_id}] Найдено файлов для обработки: {total}")

        # 3) Инициализировать детектор и эмбеддер
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 4) Извлечь эмбеддинг образца
        sample_img = cv2.imread(face_path)
        faces = detector.detectMultiScale(sample_img, scaleFactor=1.1, minNeighbors=4)
        if len(faces) == 0:
            raise RuntimeError("На фото-образце не найдено лицо")
        x, y, w, h = faces[0]
        sample_face = cv2.resize(sample_img[y:y + h, x:x + w], (160, 160))
        sample_emb = embedder.embeddings([sample_face])[0]
        logging.info(f"[{job_id}] Эталонный эмбеддинг получен")

        # 5) Перебор и сравнение
        found = []
        for idx, fname in enumerate(files, start=1):
            tasks[job_id]['processed'] = idx
            logging.info(f"[{job_id}] Обработка {idx}/{total}: {fname}")

            img = cv2.imread(os.path.join(img_dir, fname))
            if img is None:
                logging.warning(f"[{job_id}] Не удалось прочитать {fname}, пропускаем")
                continue
            faces = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
            if not len(faces):
                continue

            x, y, w, h = faces[0]
            face = cv2.resize(img[y:y + h, x:x + w], (160, 160))
            emb = embedder.embeddings([face])[0]
            # косинусное сходство
            cos_sim = np.dot(sample_emb, emb) / (np.linalg.norm(sample_emb) * np.linalg.norm(emb))
            if cos_sim >= 0.5:
                found.append(fname)
                logging.info(f"[{job_id}] Совпадение: {fname} (sim={cos_sim:.2f})")

        # 6) Упаковать найденное
        tasks[job_id]['found'] = found
        tasks[job_id]['status'] = 'done'
        logging.info(f"[{job_id}] Задача завершена. Найдено {len(found)}/{total}")

    except Exception as e:
        logging.exception(f"[{job_id}] Ошибка в задаче:")
        tasks[job_id]['status'] = 'error'
        tasks[job_id]['error'] = str(e)

# --- Маршруты --------------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        zip_file = request.files.get('zip_file')
        face_image = request.files.get('face_image')
        if not zip_file or not face_image:
            logging.error("POST без необходимых файлов")
            return 'Оба файла (zip и фото) должны быть загружены.', 400

        job_id = str(uuid.uuid4())
        tasks[job_id] = {'status': 'processing', 'processed': 0, 'total': 0}
        zip_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{secure_filename(zip_file.filename)}")
        face_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{secure_filename(face_image.filename)}")
        zip_file.save(zip_path)
        face_image.save(face_path)

        # Запуск фонового потока
        logging.info(f"[{job_id}] Получены файлы, запускаем фоновый поток")
        threading.Thread(target=process_task, args=(job_id, zip_path, face_path), daemon=True).start()
        return redirect(url_for('status', job_id=job_id))
    return render_template('index.html')

@app.route('/status/<job_id>')
def status(job_id):
    task = tasks.get(job_id)
    if not task:
        return 'Неверный идентификатор задачи.', 404
    return render_template('status.html',
                           job_id=job_id,
                           status=task['status'],
                           total=task['total'],
                           processed=task['processed'],
                           message=task.get('message', ''))

@app.route('/download/<job_id>')
def download(job_id):
    task = tasks.get(job_id)
    if not task or task.get('status') != 'done':
        return 'Результат ещё недоступен.', 404
    return send_file(task['result_path'], as_attachment=True, download_name='found_faces.zip')

# --- Запуск приложения ---------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)