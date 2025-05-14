FROM python:3.11-slim

WORKDIR /app

# зависимости
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

CMD ["python3", "app.py"]