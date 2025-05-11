import os
import io
import zipfile
import tempfile
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import keras
from mtcnn.mtcnn import MTCNN
import numpy as np
from PIL import Image

# Инициализация Flask
app = Flask(__name__)

# Путь для временного хранения
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Загрузка моделей
detector = MTCNN()
# Предобученная модель FaceNet в папке model/facenet_keras.h5
embedding_model = keras.models.load_model('model/facenet_keras.h5')

# Порог сходства (минимальное значение cosine similarity)
THRESHOLD = 0.5

# Функция для извлечения лица и приведения к нужному размеру
def extract_face(image_path, required_size=(160, 160)):
    image = Image.open(image_path)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    results = detector.detect_faces(pixels)
    if not results:
        return None
    # Берём первый найденный фрагмент
    x1, y1, width, height = results[0]['box']
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

# Преобразуем лицо в embedding
def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = embedding_model.predict(samples)
    return yhat[0]

# Сравнение: cosine similarity
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Сохранение файлов
        zip_file = request.files['zip_file']
        face_image = request.files['face_image']
        zip_path = os.path.join(UPLOAD_FOLDER, secure_filename(zip_file.filename))
        face_path = os.path.join(UPLOAD_FOLDER, secure_filename(face_image.filename))
        zip_file.save(zip_path)
        face_image.save(face_path)

        # Распаковка zip
        extract_dir = tempfile.mkdtemp(dir=UPLOAD_FOLDER)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Получаем embedding образца
        sample_face = extract_face(face_path)
        if sample_face is None:
            return "Не удалось обнаружить лицо на образце"
        sample_emb = get_embedding(sample_face)

        # Перебираем изображения
        matches = []
        for root, _, files in os.walk(extract_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    face = extract_face(fpath)
                    if face is None:
                        continue
                    emb = get_embedding(face)
                    sim = cosine_similarity(sample_emb, emb)
                    if sim >= THRESHOLD:
                        matches.append(fpath)
                except Exception:
                    continue

        # Формируем zip с найденными
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w') as zip_out:
            for match in matches:
                zip_out.write(match, arcname=os.path.basename(match))
        output.seek(0)

        return send_file(
            output,
            mimetype='application/zip',
            as_attachment=True,
            download_name='found_faces.zip'
        )

    return render_template('templates/index.html')

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)