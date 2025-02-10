from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
import os
import imutils

import tensorflow as tf
from werkzeug.utils import secure_filename
from datetime import datetime
# cnn model
#model = tf.keras.models.load_model('model/BangIso_v7.h5')
model = tf.keras.models.load_model('model/Ekush_7feb.h5')

#bangla charcter for ekush dataset
characters_list = ['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ', 
    'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ', 
    'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 
    'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 
    'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 
    'হ', 'ড়', 'ঢ়', 'য়', 'ৎ', 'ং', 'ঃ', 'ঁ',
    'ব্দ', 'ঙ্গ', 'স্ক', 'স্ফ', 'চ্ছ', 'স্থ', 'ক্ত', 'স্ন', 'ষ্ণ', 'ম্প', 'প্ত', 'ম্ব', 'ত্থ', 'দ্ভ', 'ষ্ঠ', 
    'ল্প', 'ষ্প', 'ন্দ', 'ন্ধ', 'স্ম', 'ণ্ঠ', 'স্ত', 'ষ্ট', 'ন্ম', 'ত্ত', 'ঙ্খ', 'ত্ন', 'ন্ড', 'জ্ঞ', 'ড্ড', 
    'ক্ষ', 'দ্ব', 'চ্চ', 'ক্র', 'দ্দ', 'জ্জ', 'ক্ক', 'ন্ত', 'ক্ট', 'ঞ্চ', 'ট্ট', 'শ্চ', 'ক্স', 'জ্ব', 'ঞ্জ', 
    'দ্ধ', 'ন্ন', 'ঘ্ন', 'ক্ল', 'হ্ন', 'ল্ত', 'স্প',
    '০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯'
]
app = Flask(__name__)
UPLOAD_FOLDER = os.path.abspath('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    new_w = 600
    new_h = int((new_w / w) * h)
    img_resized = cv2.resize(img, (new_w, new_h))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 21)
    invertion = 255 - adaptive
    dilation = cv2.dilate(invertion, np.ones((3,3)))
    edges = cv2.Canny(dilation, 50, 200)
    dilation = cv2.dilate(edges, np.ones((3,3)))
    return img_resized, gray, dilation


def draw_prediction(img, x, y, w, h, character):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 128, 0), 2)




def find_contours(img):
    conts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    return sorted(conts, key=lambda c: cv2.boundingRect(c)[0])

def extract_roi(img, x, y, w, h, margin=2):
    return img[y - margin:y + h , x - margin:x + w + margin]

def process_box(gray, x, y, w, h):
    roi = extract_roi(gray, x, y, w, h, margin=2)
    thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    resized = cv2.resize(thresh, (28, 28))
    normalized = resized.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=-1)
@app.route("/about")
def about():
    
    return render_template("about.html")
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process image
            img_resized, gray, processed_img = preprocess_image(file_path)
            contours = find_contours(processed_img)

            # ROIs and boxes
            characters = []
            boxes = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if 4 <= w <= 160 and 12 <= h <= 140:
                    roi = process_box(gray, x, y, w, h)
                    characters.append(roi)
                    boxes.append((x, y, w, h))

            # Predict characters
            if characters:
                pixels = np.array(characters, dtype='float32')
                predictions = model.predict(pixels)
                predicted_text = "".join(characters_list[np.argmax(pred)] for pred in predictions)
            else:
                predicted_text = "No characters detected"

            # annotated image
            img_with_boxes = img_resized.copy()
            for pred, (x, y, w, h) in zip(predictions, boxes):
                character = characters_list[np.argmax(pred)]
                draw_prediction(img_with_boxes, x, y, w, h, character)

            # Save 
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            annotated_filename = f"annotated_{timestamp}_{filename}"
            annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
            
            # Convert 
            img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
            cv2.imwrite(annotated_path, img_with_boxes_rgb)

            return render_template("index.html", 
                                 text=predicted_text, 
                                 filename=filename,
                                 annotated_filename=annotated_filename)
    
    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
