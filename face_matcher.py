import numpy as np
import cv2
from keras_facenet import FaceNet

# логика для сравнения лиц

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
embedder = FaceNet()
THRESHOLD = 0.5

def detect_faces(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    return img, faces

def extract_embedding(image_bytes):
    img, faces = detect_faces(image_bytes)
    if faces is None or len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    crop = cv2.resize(img[y:y+h, x:x+w], (160, 160))
    emb = embedder.embeddings([crop])[0]
    return emb

def find_matches(zip_bytes, sample_emb, callback):
    import zipfile
    import io

    matches = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(('jpg', 'jpeg', 'png'))]
        already_check = 0
        callback(0, len(names))
        for name in names:
            arr = np.frombuffer(z.read(name), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            faces = face_cascade.detectMultiScale(img, 1.1, 4)
            for (x, y, w, h) in faces:
                face = cv2.resize(img[y:y+h, x:x+w], (160, 160))
                emb = embedder.embeddings([face])[0]
                sim = np.dot(sample_emb, emb) / (np.linalg.norm(sample_emb) * np.linalg.norm(emb))
                if sim >= THRESHOLD:
                    matches.append((name, z.read(name)))
                    break
            already_check +=1
            callback(already_check, len(names))
    return matches, len(names)