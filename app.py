from flask import Flask
from routes import setup_routes

# Точка входа

app = Flask(__name__)
setup_routes(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)