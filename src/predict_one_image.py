
from predict import load_eye_model, predict_eye_disease

# Load model once
model = load_eye_model()

# Predict an eye image
result = predict_eye_disease("dataset/cataract/11.jpg", model)

print("Disease:", result["predicted_class"])
print("Confidence:", f"{result['confidence'] * 100:.2f}%")
print("Probabilities:", result["probabilities"])
