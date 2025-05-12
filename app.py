from flask import Flask, request, render_template
import zipfile
import io

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # получаем файл из формы
        zip_file = request.files.get('zip_file')
        if not zip_file:
            return "Нет файла ZIP", 400

        # читаем содержимое ZIP из памяти
        data = zip_file.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            # считаем только реальные файлы (не папки)
            count = sum(1 for name in z.namelist() if not name.endswith('/'))

        # рендерим result.html и передаём count
        return render_template('result.html', count=count)

    # GET — отдать форму
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)