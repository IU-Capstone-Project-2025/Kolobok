from flask import Flask, request, jsonify, send_from_directory, render_template, url_for, redirect
from flask_cors import CORS
import os
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

CHAT_FILE = 'chat_history.json'
TIRES_DIR = 'tires'
os.makedirs(TIRES_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/api/messages', methods=['GET', 'POST'])
def messages():
    if request.method == 'GET':
        if os.path.exists(CHAT_FILE):
            with open(CHAT_FILE, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        return jsonify([])
    else:
        data = request.json
        with open(CHAT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return jsonify({'status': 'ok'})

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(TIRES_DIR, filename)
    file.save(filepath)
    return jsonify({'url': f'/tires/{filename}'})

@app.route('/tires/<filename>')
def get_tire(filename):
    return send_from_directory(TIRES_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True) 