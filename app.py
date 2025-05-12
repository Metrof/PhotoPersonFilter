import os
import zipfile
import tempfile
import threading
import uuid
from flask import Flask, render_template, request, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from keras_facenet import FaceNet
import cv2
from cv2 import data

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
    tasks[job_id] = {'status': 'processing', 'total': 0, 'processed': 0}
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Распаковка архива
            extract_dir = os.path.join(tmpdir, 'photos')
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

            # Подсчет общего количества файлов
            total_files = sum(len(files) for _, _, files in os.walk(extract_dir))
            tasks[job_id]['total'] = total_files

            # Обработка фото-образца
            sample_face = extract_face(face_path)
            if sample_face is None:
                raise ValueError('Лицо на образце не найдено')
            sample_emb = embedder.embeddings([sample_face])[0]

            matches = []
            # Перебор и сравнение
            for root, _, files in os.walk(extract_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    face = extract_face(fpath)
                    # Обновляем счётчик обработанных
                    tasks[job_id]['processed'] += 1
                    if face is None:
                        continue
                    emb = embedder.embeddings([face])[0]
                    if cosine_similarity(sample_emb, emb) >= THRESHOLD:
                        matches.append(fpath)

        # Упаковка найденных в zip
        result_path = os.path.join(RESULT_FOLDER, f"{job_id}.zip")
        with zipfile.ZipFile(result_path, 'w', zipfile.ZIP_DEFLATED) as zf_out:
            for m in matches:
                zf_out.write(m, arcname=os.path.basename(m))

        tasks[job_id].update({'status': 'done', 'result_path': result_path})
    except Exception as e:
        tasks[job_id].update({'status': 'error', 'message': str(e)})

# --- Маршруты --------------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        zip_file = request.files.get('zip_file')
        face_image = request.files.get('face_image')
        if not zip_file or not face_image:
            return 'Оба файла (zip и фото) должны быть загружены.', 400

        job_id = str(uuid.uuid4())
        zip_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{secure_filename(zip_file.filename)}")
        face_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{secure_filename(face_image.filename)}")
        zip_file.save(zip_path)
        face_image.save(face_path)

        # Запуск фонового потока
        thread = threading.Thread(target=process_task, args=(job_id, zip_path, face_path), daemon=True)
        thread.start()

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