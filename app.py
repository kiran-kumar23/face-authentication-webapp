# app.py
import cv2
import io
import os
import torch
import base64
import sqlite3
import numpy as np
import threading
from PIL import Image
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from facenet_pytorch import MTCNN, InceptionResnetV1

# ------------------- Flask App -------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super-secret-key")

# ------------------- Models -------------------
mtcnn = MTCNN(keep_all=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# ------------------- DB Setup -------------------
DB_PATH = 'users.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        faiss_id INTEGER
    )''')
    conn.commit()
    conn.close()

init_db()

# ------------------- FAISS Setup -------------------
import faiss
EMBED_DIM = 512
index = faiss.IndexFlatL2(EMBED_DIM)
faiss_id_to_user = []
faiss_lock = threading.Lock()

# ------------------- Helpers -------------------
def image_from_b64(b64str):
    header, encoded = b64str.split(',', 1) if ',' in b64str else (None, b64str)
    b = base64.b64decode(encoded)
    return Image.open(io.BytesIO(b)).convert('RGB')

def get_embedding_from_pil(pil_img):
    face = mtcnn(pil_img)
    if face is None:
        return None
    with torch.no_grad():
        emb = resnet(face.unsqueeze(0))
    return emb[0].numpy()

def login_required(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please login first.")
            return redirect(url_for('home'))
        return func(*args, **kwargs)
    return wrapper

# ------------------- Routes -------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/faceauth')
def faceauth():
    return render_template('index.html')

@app.route('/profile')
@login_required
def profile_page():
    user_data = {
        "user_id": session.get("user_id"),
        "username": session.get("username")
    }
    return render_template('profile.html', user=user_data)

@app.route('/liveness_check', methods=['POST'])
def liveness_check():
    data = request.get_json()
    frames = data.get('frames', [])

    blink_count = 0
    smile_detected = False

    # Load Haar cascades
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    prev_eyes_detected = True  # track across frames

    for f in frames:
        # Decode frame
        img_data = np.frombuffer(base64.b64decode(f.split(',')[1]), np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Detect eyes ---
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # --- Detect smile ---
        smiles = smile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.8,   # stricter
            minNeighbors=25,
            minSize=(25, 25)
        )

        # Blink detection: eyes disappear then reappear
        if prev_eyes_detected and len(eyes) == 0:
            prev_eyes_detected = False
        elif not prev_eyes_detected and len(eyes) >= 1:
            blink_count += 1
            prev_eyes_detected = True

        # Smile detection
        if len(smiles) > 0:
            smile_detected = True

    print(f"DEBUG: Blinks={blink_count}, Smile={smile_detected}")

    # ✅ Liveness passes if either condition is true
    if blink_count >= 1 or smile_detected:
        return jsonify(success=True)

    return jsonify(success=False)



# ------------------- Signup -------------------
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get('username')
    images = data.get('images')

    if not username or not images:
        return jsonify({'error': 'username and images required'}), 400

    embeddings = []
    for b64 in images:
        pil = image_from_b64(b64)
        emb = get_embedding_from_pil(pil)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        return jsonify({'error': 'no faces detected'}), 400

    mean_emb = np.mean(np.stack(embeddings), axis=0).astype('float32')

    with faiss_lock:
        if index.ntotal > 0:
            D, I = index.search(np.expand_dims(mean_emb, axis=0), 1)
            if D[0][0] <= 0.9:
                existing_user_id = faiss_id_to_user[int(I[0][0])]
                return jsonify({'error': 'Face already registered', 'user_id': existing_user_id}), 400

        faiss_id = len(faiss_id_to_user)
        index.add(np.expand_dims(mean_emb, axis=0))
        faiss_id_to_user.append(None)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, faiss_id) VALUES (?, ?)', (username, faiss_id))
        user_id = c.lastrowid
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'error': 'Username already exists'}), 400
    conn.close()

    faiss_id_to_user[faiss_id] = user_id

    session['user_id'] = user_id
    session['username'] = username

    return jsonify({'status': 'ok', 'user_id': user_id})

# ------------------- Login Face -------------------
@app.route('/login_face', methods=['POST'])
def login_face():
    data = request.json
    image = data.get('image')
    if not image:
        return jsonify({'error': 'image required'}), 400

    pil = image_from_b64(image)
    emb = get_embedding_from_pil(pil)
    if emb is None:
        return jsonify({'error': 'no face detected'}), 400

    emb = emb.astype('float32').reshape(1, -1)

    with faiss_lock:
        if index.ntotal == 0:
            return jsonify({'error': 'no enrolled users'}), 400
        D, I = index.search(emb, 1)

    best_dist = float(D[0][0])
    best_idx = int(I[0][0])
    THRESHOLD = 0.9

    if best_dist <= THRESHOLD:
        user_id = faiss_id_to_user[best_idx]
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT username FROM users WHERE id=?", (user_id,))
        row = c.fetchone()
        conn.close()
        username = row[0] if row else "Unknown"

        session['user_id'] = user_id
        session['username'] = username

        return jsonify({'status': 'ok', 'user_id': user_id, 'username': username})
    else:
        return jsonify({'status': 'no_match', 'distance': best_dist}), 401

# ------------------- Whoami -------------------
@app.route('/whoami', methods=['GET'])
@login_required
def whoami():
    return jsonify({'user_id': session.get('user_id'), 'username': session.get('username')})

# ------------------- Logout -------------------
@app.route('/logout')
@login_required
def logout():
    session.clear()
    return redirect(url_for('home'))

# ------------------- Run -------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
