import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


# Must match the class_names argument used during training.
CLASS_NAMES = ["normal", "cataract", "conjunctivitis", "pterygium"]
IMAGE_SIZE = (224, 224)

# src/predict.py -> project root -> models/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_efficientnetb0_finetuned.keras"


def load_eye_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    """Load the saved model for inference (no optimizer required)."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(str(model_path), compile=False)
    return model


def preprocess_image(image_path: str | Path) -> np.ndarray:
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = tf.keras.utils.load_img(
        str(image_path),
        color_mode="rgb",
        target_size=IMAGE_SIZE,
        interpolation="bilinear",
    )
    image_array = tf.keras.utils.img_to_array(image, dtype="float32")
    return np.expand_dims(image_array, axis=0)  # (1, 224, 224, 3)


def predict_eye_disease(image_path: str | Path, model: tf.keras.Model) -> dict:

    image_batch = preprocess_image(image_path)
    outputs = model.predict(image_batch, verbose=0)
    scores = np.asarray(outputs)[0]

    if scores.shape != (len(CLASS_NAMES),):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} output probabilities, got shape {scores.shape}."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("Model returned invalid prediction scores.")

    predicted_index = int(np.argmax(scores))
    probabilities = {
        name: round(float(score), 6)
        for name, score in zip(CLASS_NAMES, scores)
    }
    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": round(float(scores[predicted_index]), 6),
        "probabilities": probabilities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify an eye image using EfficientNetB0")
    parser.add_argument("image", type=Path, help="Path to an eye image (JPG, JPEG, PNG, etc.)")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of formatted text")
    args = parser.parse_args()

    model = load_eye_model(args.model)
    result = predict_eye_disease(args.image, model)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\nEYE DISEASE CLASSIFICATION")
        print("-" * 35)
        print(f"Image: {args.image}")
        print(f"Prediction: {result['predicted_class']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}%")
        print("\nClass probabilities:")
        for name, probability in sorted(
            result["probabilities"].items(), key=lambda item: item[1], reverse=True
        ):
            print(f"  {name:<16} {probability * 100:6.2f}%")
        print("\nFor research use only; not a medical diagnosis.")


if __name__ == "__main__":
    main()