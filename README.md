# 👁️ Eye Disease Detection System (OcuSight AI)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00.svg?logo=tensorflow)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg?logo=flask)](https://flask.palletsprojects.com/)
[![Accuracy](https://img.shields.io/badge/Top%20Test%20Accuracy-93.58%25-brightgreen.svg)]()
[![Model](https://img.shields.io/badge/Production%20Model-Fine--Tuned%20EfficientNetB0-purple.svg)]()

An end-to-end deep learning computer vision system for multi-class classification of ocular conditions from photographic eye images. The project encompasses rigorous comparative experimentation across three distinct neural architectures, model fine-tuning, automated stratified dataset preparation, a modular Python inference engine, and **OcuSight AI**—a modern, privacy-focused Flask web application.

---

## 📌 Table of Contents

- [Overview & Medical Conditions](#-overview--medical-conditions)
- [Key Features](#-key-features)
- [Model Performance & Comparison](#-model-performance--comparison)
- [Dataset Architecture](#-dataset-architecture)
- [Repository Structure](#-repository-structure)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Web Application (OcuSight AI)](#-web-application-ocusight-ai)
- [REST API Specification](#-rest-api-specification)
- [Command-Line & Programmatic Usage](#-command-line--programmatic-usage)
- [Model Training & Research Notebooks](#-model-training--research-notebooks)
- [Clinical & Ethical Disclaimer](#-clinical--ethical-disclaimer)

---

## 🔬 Overview & Medical Conditions

The system classifies eye photographs into four mutually exclusive diagnostic categories:

| Condition | Visual & Clinical Description | Clinical Significance |
| :--- | :--- | :--- |
| **Normal** | Clear cornea, healthy sclera, transparent anterior chamber, no visible lesions. | Reference baseline; absence of target pathologies in the study scope. |
| **Cataract** | Opacification and clouding of the eye's natural crystalline lens. | Leading cause of reversible blindness worldwide. |
| **Conjunctivitis** | Hyperemia, vascular dilation, and inflammation of the conjunctival tissue ("pink eye"). | Highly transmissible infectious or allergic irritation requiring prompt triage. |
| **Pterygium** | Non-cancerous, triangular wedge-shaped fibrovascular growth over cornea and sclera ("surfer's eye"). | Progressive growth may cause astigmatism and visual obstruction if untreated. |

---

## ⚡ Key Features

- **Architectural Benchmark**: Systematic evaluation of a custom Residual CNN (mini-ResNet), MobileNetV2, and EfficientNetB0 under identical training pipelines.
- **State-of-the-Art Accuracy**: Top-performing model achieves **93.58% test accuracy** and **0.2250 test loss** with balanced F1-scores (>91% across all classes).
- **Two-Stage Fine-Tuning**: ImageNet pre-training followed by frozen-base feature extraction and targeted top-layer fine-tuning with conservative learning rates ($1 \times 10^{-5}$).
- **Modern Web Interface (OcuSight AI)**: Responsive glassmorphic UI featuring drag-and-drop file upload, client-side validation, animated confidence dials, and class probability distributions.
- **Privacy-Preserving Inference**: Images are processed temporarily in volatile memory (`tempfile.TemporaryDirectory`) and permanently deleted immediately after prediction.
- **Flexible Deployment Interfaces**: Supports interactive web sessions, headless JSON REST API requests, and CLI batch predictions.

---

## 📊 Model Performance & Comparison

Three distinct architectures were developed, trained, and tested on an independent 15% stratified test split (623 images):

### 1. Benchmark Summary

| Model Architecture | Parameters | Epochs | Val Accuracy | Test Accuracy | Test Loss | Key Strengths / Characteristics |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Custom mini-ResNet** | ~2.49M | 30 | 77.53% | **87.00%** | 0.7960 | Built from scratch with custom `ResidualUnit` layers, He-normal initialization, and label smoothing (0.1). |
| **MobileNetV2 (Fine-Tuned)** | ~2.26M | 20 + 10 | 90.85% | **87.48%** | 0.3074 | Ultra-lightweight footprint; unfreezing top 30 layers; excellent inference latency for edge devices. |
| **EfficientNetB0 (Fine-Tuned)** ⭐ | ~4.21M | 15 + 15 | **95.02%** | **93.58%** | **0.2250** | **Best overall performer.** Compound scaling, unfreezing top 20 layers, superior precision and recall balance. |

⭐ *Selected as the production model (`models/best_efficientnetb0_finetuned.keras`).*

### 2. Detailed Evaluation: Winning Model (Fine-Tuned EfficientNetB0)

Evaluated on the 623-image holdout test dataset:

```text
                  precision    recall  f1-score   support

          normal     0.9600    0.9091    0.9339       132
        cataract     0.9659    0.9605    0.9632       177
  conjunctivitis     0.9235    0.9330    0.9282       194
       pterygium     0.8889    0.9333    0.9106       120

        accuracy                         0.9358       623
       macro avg     0.9346    0.9340    0.9339       623
    weighted avg     0.9366    0.9358    0.9359       623
```

---

## 🗂️ Dataset Architecture

The dataset comprises **4,151 expert-labeled eye photographs** partitioned into stratified training, validation, and test subsets to maintain exact class proportions:

```text
dataset/ (4,151 total)
├── cataract/        (1,176 images)
├── conjunctivitis/  (1,293 images)
├── normal/          (882 images)
├── pterygium/       (800 images)
└── labels.csv       (tabular index with metadata)
```

### Stratified Distribution (70% / 15% / 15%)

| Class Name | Class Index | Total Images | Train Split (70%) | Validation Split (15%) | Test Split (15%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal** | `0` | 882 | 617 | 133 | 132 |
| **Cataract** | `1` | 1,176 | 823 | 176 | 177 |
| **Conjunctivitis** | `2` | 1,293 | 905 | 194 | 194 |
| **Pterygium** | `3` | 800 | 560 | 120 | 120 |
| **Total** | — | **4,151** | **2,905** | **623** | **623** |

### Data Augmentation Pipeline

To improve generalization and mitigate overfitting, online data augmentation is applied during training:
- **Random Horizontal Flip**: `RandomFlip("horizontal")`
- **Subtle Rotation**: `RandomRotation(factor=0.08)`
- **Dynamic Zooming**: `RandomZoom(height_factor=(-0.15, 0.10))`
- **Translations**: `RandomTranslation(height_factor=0.05, width_factor=0.05)`
- **Color & Contrast**: `RandomContrast(factor=0.15)` & `RandomBrightness(factor=0.15)`

---

## 📂 Repository Structure

```text
Eye-Disease-Detection-system/
├── dataset/                               # Raw labeled eye image collection
│   ├── cataract/                          # Cataract eye photographs
│   ├── conjunctivitis/                    # Conjunctivitis eye photographs
│   ├── normal/                            # Healthy control eye photographs
│   ├── pterygium/                         # Pterygium eye photographs
│   └── labels.csv                         # Full dataset index and ground truth
│
├── split_dataset/                         # Stratified train/val/test splits (70/15/15)
│   ├── train/                             # 2,905 training images
│   ├── val/                               # 623 validation images
│   └── test/                              # 623 test images
│
├── models/                                # Saved serialized Keras models (.keras)
│   ├── best_efficientnetb0.keras          # EfficientNetB0 feature extractor
│   ├── best_efficientnetb0_finetuned.keras# Winning production model (93.58% acc)
│   ├── best_eye_disease_model_custom_model.keras # Custom ResNet baseline
│   ├── best_mobilenetv2.keras             # MobileNetV2 feature extractor
│   └── best_mobilenetv2_finetuned.keras   # MobileNetV2 fine-tuned model
│
├── notebooks/                             # Jupyter notebooks for experimentation
│   ├── label data.ipynb                   # Dataset audit, EDA & stratified splitting
│   ├── experiment 1_custom model.ipynb    # Custom ResNet design & training
│   ├── experimnet 2_EfficientNetBO.ipynb  # EfficientNetB0 2-stage transfer learning
│   └── experiment 3_MobileNetV2.ipynb     # MobileNetV2 transfer learning & tuning
│
├── results/                               # Test evaluations and training histories
│   ├── experiment2_test_results.csv       # EfficientNetB0 test metrics
│   ├── test_results.csv                   # Custom ResNet test metrics
│   ├── training_history.csv               # Custom model epoch training logs
│   ├── mobilenetv2_history.csv            # MobileNetV2 initial training logs
│   └── Tuned_mobilenetv2_history.csv      # MobileNetV2 fine-tuning logs
│
├── src/                                   # Core inference and prediction modules
│   ├── __init__.py                        # Package marker
│   ├── predict.py                         # Production inference CLI & API utilities
│   └── predict_one_image.py               # Minimal prediction demonstration script
│
├── static/                                # Web application static assets
│   ├── css/style.css                      # OcuSight AI responsive custom styling
│   └── js/app.js                          # Drag-and-drop & API async handler
│
├── templates/                             # Flask HTML templates
│   └── index.html                         # OcuSight AI single-page frontend
│
├── app.py                                 # Flask backend server & REST API
├── requirements.txt                       # Consolidated Python dependencies
├── requirements-web.txt                   # Web-specific dependencies
└── README.md                              # Comprehensive project documentation
```

---

## 🛠️ Installation & Environment Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12+ (tested with Python 3.14)
- Pip package manager
- Optional: NVIDIA GPU with CUDA drivers for accelerated training

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/Eye-Disease-Detection-system.git
cd Eye-Disease-Detection-system
```

### 3. Create and Activate a Virtual Environment
- **On Linux/macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🌐 Web Application (OcuSight AI)

The project includes an interactive web platform built with Flask, Bootstrap 5, and vanilla JavaScript.

### Running the Web Server
Execute from the project root:
```bash
python app.py
```

The application will start locally at:
```text
http://127.0.0.1:5000
```

### UI Capabilities:
1. **Interactive Drag & Drop**: Drop JPG or PNG images directly into the browser.
2. **Instant Preview & Format Validation**: Enforces maximum upload size (8 MB) and file integrity check using Pillow.
3. **Animated Confidence Dial**: Circular SVG gauge visually rendering prediction certainty.
4. **Multi-Class Breakdown**: Real-time percentage bars for all 4 diagnostic classes.
5. **Session Isolation**: Each upload is processed in a temporary directory and automatically deleted upon completion.

---

## 🔌 REST API Specification

The Flask server provides headless programmatic endpoints for automated pipelines or mobile clients.

### 1. Health Check Endpoint
```http
GET /health
```
**Response:**
```json
{
  "status": "ok",
  "model_file_exists": true
}
```

### 2. Disease Prediction Endpoint
```http
POST /predict
Content-Type: multipart/form-data
```

**Parameters:**
| Parameter | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `image` | File (Binary) | Yes | Image file in JPG, JPEG, or PNG format (<8 MB). |

**Example using cURL:**
```bash
curl -X POST -F "image=@dataset/cataract/11.jpg" http://127.0.0.1:5000/predict
```

**Example JSON Response:**
```json
{
  "success": true,
  "prediction": {
    "label": "cataract",
    "display_name": "Cataract",
    "confidence": 98.42,
    "summary": "The model selected 'Cataract', a condition associated with clouding of the eye's natural lens.",
    "probabilities": [
      { "label": "normal", "display_name": "Normal", "percentage": 0.85 },
      { "label": "cataract", "display_name": "Cataract", "percentage": 98.42 },
      { "label": "conjunctivitis", "display_name": "Conjunctivitis", "percentage": 0.41 },
      { "label": "pterygium", "display_name": "Pterygium", "percentage": 0.32 }
    ],
    "disclaimer": "AI-generated class prediction for educational/research use, not a medical diagnosis."
  }
}
```

---

## 💻 Command-Line & Programmatic Usage

### 1. Command-Line Interface (CLI)

Classify any ocular image directly from your terminal using `src/predict.py`:

```bash
# Standard formatted summary
python src/predict.py dataset/cataract/11.jpg

# JSON-formatted output
python src/predict.py dataset/cataract/11.jpg --json

# Using an alternative trained model
python src/predict.py dataset/cataract/11.jpg --model models/best_mobilenetv2_finetuned.keras
```

**Example CLI Output:**
```text
EYE DISEASE CLASSIFICATION
-----------------------------------
Image: dataset/cataract/11.jpg
Prediction: cataract
Confidence: 98.42%

Class probabilities:
  cataract          98.42%
  normal             0.85%
  conjunctivitis     0.41%
  pterygium          0.32%

For research use only; not a medical diagnosis.
```

### 2. Python API Integration

Import the trained model directly in your Python code:

```python
from pathlib import Path
from src.predict import load_eye_model, predict_eye_disease

# 1. Load model once (caches compiled graph)
model = load_eye_model(Path("models/best_efficientnetb0_finetuned.keras"))

# 2. Run inference on target image
image_path = "dataset/cataract/11.jpg"
result = predict_eye_disease(image_path, model)

print(f"Predicted Class: {result['predicted_class']}")
print(f"Confidence: {result['confidence'] * 100:.2f}%")
print(f"Full Probabilities: {result['probabilities']}")
```

---

## 📓 Model Training & Research Notebooks

To reproduce the experiments or train models from scratch, run the notebooks in order:

1. **`notebooks/label data.ipynb`**:
   - Parses the image dataset directory.
   - Generates tabular metadata and verifies class balances.
   - Performs a 70/15/15 stratified split into `split_dataset/train`, `split_dataset/val`, and `split_dataset/test`.

2. **`notebooks/experiment 1_custom model.ipynb`**:
   - Implements a custom convolutional neural network with residual shortcut connections (`ResidualUnit`).
   - Uses He-normal initialization, Batch Normalization, and L2 regularization.
   - Evaluates baseline accuracy (87.00% test accuracy).

3. **`notebooks/experimnet 2_EfficientNetBO.ipynb`**:
   - Transfer learning using ImageNet weights on EfficientNetB0.
   - Phase 1: Freezes base layers and trains custom classification head (15 epochs).
   - Phase 2: Unfreezes top 20 convolutional layers for fine-tuning at $lr = 1 \times 10^{-5}$ (15 epochs).
   - Generates test confusion matrices and classification reports (**93.58% test accuracy**).

4. **`notebooks/experiment 3_MobileNetV2.ipynb`**:
   - Transfer learning using MobileNetV2 for edge-optimized deployment.
   - Evaluates performance trade-offs against EfficientNetB0 (87.48% test accuracy).

---

## ⚠️ Clinical & Ethical Disclaimer

> **IMPORTANT NOTICE:**  
> This software is an **experimental research and educational tool** designed to demonstrate deep learning applications in computer vision. It has **NOT** been approved, cleared, or certified as a medical device or diagnostic system by the U.S. FDA, EMA, or any national health authority.  
>  
> - Predictions generated by this system must **NEVER** replace clinical examination, professional medical consultation, or specialized ophthalmological diagnosis.
> - The software is provided "as is", without warranty of any kind.
> - Anyone experiencing eye discomfort, irritation, visual disturbance, or sudden vision loss should consult a licensed ophthalmologist or seek emergency medical care immediately.

---
