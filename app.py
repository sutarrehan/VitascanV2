from flask import Flask, render_template, request
import os, base64, uuid, datetime, time
import numpy as np
import cv2

from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from utils import is_face_present

# PDF
from reportlab.pdfgen import canvas

# ---------------- SETUP ----------------
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/reports', exist_ok=True)

ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}

# ---------------- MODEL ----------------
model = load_model('model/alexnet_model.h5')

with open("model/labels.txt") as f:
    VITAMINS = [line.strip() for line in f]

# ---------------- HELPERS ----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(227,227))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def cleanup_files(folder, hours=2):
    now = time.time()
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path) and os.stat(path).st_mtime < now - hours*3600:
            os.remove(path)

# ---------------- PDF ----------------
def generate_pdf(image_path, vitamin, accuracy):
    file = f"static/reports/report_{uuid.uuid4().hex}.pdf"
    c = canvas.Canvas(file)
    c.drawString(100,800,f"Vitamin: {vitamin}")
    c.drawString(100,780,f"Accuracy: {accuracy}%")
    c.drawImage(image_path,100,500,width=200,height=200)
    c.save()
    return file

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    cleanup_files(app.config['UPLOAD_FOLDER'])

    save_path = None

    # CAMERA
    if 'image_data' in request.form and request.form['image_data'] != '':
        image_data = request.form['image_data']
        header, encoded = image_data.split(",",1)

        filename = f"{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        with open(save_path,"wb") as f:
            f.write(base64.b64decode(encoded))

    # UPLOAD
    else:
        file = request.files.get('image')

        if not file or file.filename == '':
            return "No file uploaded"

        if not allowed_file(file.filename):
            return "Invalid file"

        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

    # FACE CHECK
    if not is_face_present(save_path):
        os.remove(save_path)
        return "Only face image allowed"

    # ---------------- AI PREDICTION ----------------
    img = preprocess_image(save_path)
    prediction = model.predict(img, verbose=0)

    # Top 3
    top_indices = prediction[0].argsort()[-3:][::-1]

    results = []
    for i in top_indices:
        results.append({
            "vitamin": VITAMINS[i],
            "confidence": round(float(prediction[0][i]) * 100, 2)
        })

    best = results[0]

    # Confidence Level
    if best["confidence"] > 85:
        level = "High"
    elif best["confidence"] > 70:
        level = "Medium"
    else:
        level = "Low"

    # PDF
    pdf_path = generate_pdf(save_path, best["vitamin"], best["confidence"])

    return render_template("result.html",
                           image_path=save_path,
                           best=best,
                           results=results,
                           level=level,
                           pdf_path=pdf_path)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)