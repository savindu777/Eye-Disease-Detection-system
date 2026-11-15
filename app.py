from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

# Your existing training-matched inference code in src/predict.py.
from src.predict import CLASS_NAMES, load_eye_model, predict_eye_disease


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_efficientnetb0_finetuned.keras"
MAX_UPLOAD_MB = 8
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_IMAGE_PIXELS = 20_000_000

DISPLAY_NAMES = {
    "normal": "Normal",
    "cataract": "Cataract",
    "conjunctivitis": "Conjunctivitis",
    "pterygium": "Pterygium",
}

CLASS_SUMMARIES = {
    "normal": "The model selected 'Normal' among its four training categories. This does not rule out other eye conditions.",
    "cataract": "The model selected 'Cataract', a condition associated with clouding of the eye's natural lens.",
    "conjunctivitis": "The model selected 'Conjunctivitis', which involves inflammation of the membrane covering the eye's surface.",
    "pterygium": "The model selected 'Pterygium', a growth of tissue on the surface of the eye.",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["JSON_SORT_KEYS"] = False


@lru_cache(maxsize=1)
def get_model():
    """Load model only on first use; reuse it across subsequent requests."""
    return load_eye_model(MODEL_PATH)


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    """Lightweight readiness endpoint (does not load TensorFlow model)."""
    return jsonify({"status": "ok", "model_file_exists": MODEL_PATH.is_file()})


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(_error):
    return jsonify({"error": f"The image must be smaller than {MAX_UPLOAD_MB} MB."}), 413


@app.post("/predict")
def predict():
    """Accept a single image, classify it, delete the temporary upload."""
    file = request.files.get("image")
    if file is None or not file.filename:
        return jsonify({"error": "Please select an eye image to upload."}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Only JPG, JPEG, and PNG images are supported."}), 400

    # Validate actual image contents, rather than trusting the filename/MIME type.
    try:
        with Image.open(file.stream) as uploaded:
            if uploaded.format not in ALLOWED_FORMATS:
                return jsonify({"error": "The uploaded file is not a JPG or PNG image."}), 400
            if uploaded.width * uploaded.height > MAX_IMAGE_PIXELS:
                return jsonify({"error": "Image dimensions are too large. Please use a smaller image."}), 400
            uploaded.verify()
        file.stream.seek(0)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return jsonify({"error": "Unable to read this image. Try a valid JPG or PNG file."}), 400

    if not MODEL_PATH.is_file():
        app.logger.error("Missing model at %s", MODEL_PATH)
        return jsonify({"error": "Trained model not found. Check the models folder on the server."}), 503

    try:
        # TemporaryDirectory automatically removes the uploaded file after use.
        with TemporaryDirectory(prefix="eye_classification_") as tmpdir:
            image_path = Path(tmpdir) / f"upload{suffix}"
            file.save(image_path)
            outcome = predict_eye_disease(image_path, get_model())
    except Exception:
        app.logger.exception("Eye-image inference failed")
        return jsonify({"error": "Prediction failed. Check the server logs and try again."}), 500

    selected = outcome["predicted_class"]
    response = {
        "label": selected,
        "display_name": DISPLAY_NAMES[selected],
        "confidence": round(float(outcome["confidence"]) * 100, 2),
        "summary": CLASS_SUMMARIES[selected],
        "probabilities": [
            {
                "label": label,
                "display_name": DISPLAY_NAMES[label],
                "percentage": round(float(outcome["probabilities"][label]) * 100, 2),
            }
            for label in CLASS_NAMES
        ],
        "disclaimer": "AI-generated class prediction for educational/research use, not a medical diagnosis.",
    }
    return jsonify({"success": True, "prediction": response})


if __name__ == "__main__":
    # Local development only. Do not enable the interactive debugger in production.
    app.run(host="127.0.0.1", port=5000, debug=False)
