from flask import Flask, request, render_template
import zipfile, io
import cv2
from cv2 import data
from keras_facenet import FaceNet

app = Flask(__name__)

# # ─── Инициализация модели и каскада (скачаются только веса при первом запуске) ───
# embedder     = FaceNet()  # автоматически скачивает и кеширует веса
# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# )
# THRESHOLD = 0.5

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # получаем файл из формы
        zip_file = request.files.get('zip_file')
        if not zip_file:
            return "Нет файла ZIP", 400

        # читаем содержимое ZIP из памяти
        zip_file_data = zip_file.read()
        with zipfile.ZipFile(io.BytesIO(zip_file_data)) as z:
            # считаем только реальные файлы (не папки)
            count = sum(1 for name in z.namelist() if not name.endswith('/'))

        # рендерим result.html и передаём count
        return render_template('result.html', count=count)

    # GET — отдать форму
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)