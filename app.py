
from flask import Flask, render_template, request
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image, UnidentifiedImageError
import numpy as np

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# Load MobileNetV2 pretrained model
model = MobileNetV2(weights="imagenet")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def classify_image(file):
    image = Image.open(file).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    image_array = preprocess_input(image_array)

    predictions = model.predict(image_array, verbose=0)

    results = decode_predictions(predictions, top=3)[0]

    output = []

    for _, label, confidence in results:
        output.append({
            "label": label.replace("_", " ").title(),
            "confidence": round(float(confidence) * 100, 2)
        })

    return output


@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            error = "Please select an image."

        elif not allowed_file(file.filename):
            error = "Upload JPG, JPEG, PNG or WEBP only."

        else:
            try:
                predictions = classify_image(file.stream)

            except (UnidentifiedImageError, OSError):
                error = "Invalid image. Please upload another image."

            except Exception:
                error = "Classification failed. Please try again."

    return render_template(
        "index.html",
        predictions=predictions,
        error=error
    )


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        error="Image size must be below 5 MB."
    ), 413


if __name__ == "__main__":
    app.run(debug=True)
