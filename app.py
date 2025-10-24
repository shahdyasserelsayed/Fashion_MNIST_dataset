from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("models/fashion_model.h5")


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L').resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    img_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(img_path)

            input_img = preprocess_image(img_path)
            preds = model.predict(input_img)[0]
            class_id = np.argmax(preds)
            prediction = CLASS_NAMES[class_id]
            confidence = round(float(preds[class_id]) * 100, 2)

    return render_template("index.html", prediction=prediction, confidence=confidence, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)
