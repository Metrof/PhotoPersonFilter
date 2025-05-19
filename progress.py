from threading import Thread, Lock
import tempfile, os, uuid, zipfile
from face_matcher import find_matches

# состояние задачи и потоки

class TaskManager:
    def __init__(self):
        self.progress = {}
        self.lock = Lock()

    def run(self, task_id, zip_bytes, sample_emb):
        with self.lock:
            self.progress[task_id] = {'current': 0, 'total': 1, 'done': False}
        thread = Thread(target=self._worker, args=(task_id, zip_bytes, sample_emb))
        thread.start()

    def _worker(self, task_id, zip_bytes, sample_emb):
        try:
            def callback(current, total_im):
                with self.lock:
                    self.progress[task_id]['current'] = current
                    self.progress[task_id]['total'] = total_im
            matches, total = find_matches(zip_bytes, sample_emb, callback)
            fname = f"found_{uuid.uuid4().hex}.zip"
            tmp_path = os.path.join(tempfile.gettempdir(), fname)
            with zipfile.ZipFile(tmp_path, 'w') as z:
                for name, data in matches:
                    z.writestr(name, data)
            with self.lock:
                self.progress[task_id].update({
                    'done': True,
                    'matches': len(matches),
                    'fname': fname,
                    'total': total,
                    'current': total
                })
        except Exception as e:
            with self.lock:
                self.progress[task_id]['done'] = True
                self.progress[task_id]['error'] = str(e)

    def get(self, task_id):
        with self.lock:
            return self.progress.get(task_id)
