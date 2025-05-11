from flask import Flask, render_template, request, redirect, send_file
import zipfile, os
from werkzeug.utils import secure_filename
from io import BytesIO

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        zip_file = request.files['zip_file']
        face_image = request.files['face_image']

        zip_path = os.path.join(UPLOAD_FOLDER, secure_filename(zip_file.filename))
        zip_file.save(zip_path)

        face_path = os.path.join(UPLOAD_FOLDER, secure_filename(face_image.filename))
        face_image.save(face_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(UPLOAD_FOLDER)

        files = os.listdir(UPLOAD_FOLDER)
        return render_template('result.html', files=files, face_image=face_image.filename)

    return render_template('index.html')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)