import os
import logging
import threading
import uuid
import tempfile
import zipfile
from io import BytesIO

from flask import Flask, request, redirect, url_for, render_template, send_file
from werkzeug.utils import secure_filename
from keras_facenet import FaceNet
import cv2
import numpy as np

# ─── Настройка логирования ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ─── Инициализация Flask и глобальных структур ────────────────────────────────
app = Flask(__name__)
tasks = {}  # job_id → { status, processed, total, found, zip_path, face_path, work_dir }
# Каскад для detecции лиц
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ─── Функция фоновой обработки ────────────────────────────────────────────────
def long_task(job_id):
    task = tasks[job_id]
    zip_path   = task['zip_path']
    face_path  = task['face_path']
    work_dir   = task['work_dir']

    try:
        logging.info(f"[{job_id}] Старт задачи: распаковка архива")
        # 1) Распаковываем ZIP
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(work_dir)
        logging.info(f"[{job_id}] Архив распакован в {work_dir}")

        # 2) Составляем список изображений
        files = [
            f for f in os.listdir(work_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        total = len(files)
        task['total'] = total
        logging.info(f"[{job_id}] Всего файлов: {total}")

        # 3) Инициализируем FaceNet-эмбеддер
        embedder = FaceNet()

        # 4) Детектим лицо на образце и получаем его эмбеддинг
        sample_img = cv2.imread(face_path)
        faces = cascade.detectMultiScale(sample_img, scaleFactor=1.1, minNeighbors=4)
        if len(faces) == 0:
            raise RuntimeError("На фото-образце не найдено лицо")
        x, y, w, h = faces[0]
        sample_face = cv2.resize(sample_img[y:y+h, x:x+w], (160,160))
        sample_emb  = embedder.embeddings([sample_face])[0]
        logging.info(f"[{job_id}] Эталонный эмбеддинг получен")

        # 5) Перебираем все файлы, сравниваем эмбеддинги
        found = []
        for idx, fname in enumerate(files, start=1):
            task['processed'] = idx
            logging.info(f"[{job_id}] Обработка {idx}/{total}: {fname}")
            img = cv2.imread(os.path.join(work_dir, fname))
            if img is None:
                logging.warning(f"[{job_id}] Не удалось загрузить {fname}, пропускаем")
                continue

            faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
            if len(faces) == 0:
                continue

            x, y, w, h = faces[0]
            face = cv2.resize(img[y:y+h, x:x+w], (160,160))
            emb  = embedder.embeddings([face])[0]
            # косинусное сходство
            sim = np.dot(sample_emb, emb) / (np.linalg.norm(sample_emb) * np.linalg.norm(emb))
            if sim >= 0.5:
                found.append(fname)
                logging.info(f"[{job_id}] Совпадение: {fname} (sim={sim:.2f})")

        # 6) Сохраняем результат и помечаем задачу как выполненную
        task['found']  = found
        task['status'] = 'done'
        logging.info(f"[{job_id}] Задача завершена: найдено {len(found)}/{total}")

    except Exception as e:
        logging.exception(f"[{job_id}] Ошибка в задаче")
        task['status'] = 'error'
        task['error']  = str(e)

# ─── Роуты ────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Проверяем наличие файлов
        if 'zip_file' not in request.files or 'face_image' not in request.files:
            logging.error("POST без необходимых файлов")
            return "Нужно загрузить и ZIP-архив, и фото-образец.", 400

        zip_f  = request.files['zip_file']
        face_f = request.files['face_image']

        # Создаём рабочую директорию
        work_dir  = tempfile.mkdtemp()
        zip_path  = os.path.join(work_dir, secure_filename(zip_f.filename))
        face_path = os.path.join(work_dir, secure_filename(face_f.filename))
        zip_f.save(zip_path)
        face_f.save(face_path)

        # Готовим запись в tasks
        job_id = str(uuid.uuid4())
        tasks[job_id] = {
            'status':   'processing',
            'processed': 0,
            'total':     0,
            'found':     [],
            'zip_path':  zip_path,
            'face_path': face_path,
            'work_dir':  work_dir
        }

        logging.info(f"[{job_id}] Файлы получены, запускаем поток")
        threading.Thread(target=long_task, args=(job_id,), daemon=True).start()

        # Перенаправляем на страницу статуса
        return redirect(url_for('status', job_id=job_id))

    return render_template('index.html')


@app.route('/status/<job_id>')
def status(job_id):
    task = tasks.get(job_id)
    if not task:
        return "Задача не найдена", 404

    if task['status'] == 'processing':
        # Страница с прогресс-баром
        return render_template(
            'status.html',
            job_id=job_id,
            processed=task['processed'],
            total=task['total']
        )
    elif task['status'] == 'done':
        # Перенаправляем на скачивание результата
        return redirect(url_for('download', job_id=job_id))
    else:
        # Ошибка
        return f"Ошибка в задаче: {task.get('error', 'неизвестная ошибка')}", 500


@app.route('/download/<job_id>')
def download(job_id):
    task = tasks.get(job_id)
    if not task or task['status'] != 'done':
        return "Результат ещё не готов", 404

    # Собираем найденные файлы в ZIP в памяти
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for fname in task['found']:
            file_path = os.path.join(task['work_dir'], fname)
            z.write(file_path, arcname=fname)
    buf.seek(0)

    return send_file(
        buf,
        as_attachment=True,
        download_name='found_faces.zip',
        mimetype='application/zip'
    )


# ─── Запуск приложения ────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
