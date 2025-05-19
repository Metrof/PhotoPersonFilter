from flask import request, render_template, redirect, url_for, send_file
from face_matcher import extract_embedding
from progress import TaskManager
import uuid

# Flask Маршруты

manager = TaskManager()

def setup_routes(app):
    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            zip_file = request.files.get('zip_file')
            face_file = request.files.get('face_image')
            if not zip_file or not face_file:
                return "ZIP и образец обязательны", 400

            img_bytes = face_file.read()
            sample_emb = extract_embedding(img_bytes)
            if sample_emb is None:
                return render_template('result.html', count=0, error="Лицо не найдено", download_url=None)

            zip_bytes = zip_file.read()
            task_id = str(uuid.uuid4())
            manager.run(task_id, zip_bytes, sample_emb)
            return redirect(url_for('status', task_id=task_id))

        return render_template('index.html')

    @app.route('/status/<task_id>')
    def status(task_id):
        task = manager.get(task_id)
        if not task:
            return "task_id не найден", 404

        if not task['done']:
            return render_template('status.html', processed=task['current'], total=task['total'], task_id=task_id, progress=task)

        return render_template('result.html', count=task['matches'], error=None, download_url=url_for('download', fname=task['fname']))

    @app.route('/download/<fname>')
    def download(fname):
        from tempfile import gettempdir
        import os
        path = os.path.join(gettempdir(), fname)
        if not os.path.exists(path):
            return "Файл не найден", 404
        return send_file(path, as_attachment=True, download_name='found_faces.zip', mimetype='application/zip')
