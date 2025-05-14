from flask import Flask, request, render_template
import zipfile, io
import cv2
from cv2 import data
from keras_facenet import FaceNet

app = Flask(__name__)

# ─── Инициализация модели и каскада (скачаются только веса при первом запуске) ───
embedder     = FaceNet()  # автоматически скачивает и кеширует веса
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
THRESHOLD = 0.5

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        zip_file = request.files.get('zip_file')
        if not zip_file:
            return "Нет ZIP-файла", 400

        # Просто считаем файлы в архиве
        data = zip_file.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            count = sum(1 for n in z.namelist() if not n.endswith('/'))

        # Здесь можно было бы сразу тестово вызвать face_cascade и embedder,
        # но пока просто переходим на страницу результата:
        return render_template('result.html', count=count)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)