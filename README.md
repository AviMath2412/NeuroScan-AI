# 🧠 NeuroScan AI — Intelligent Brain MRI Diagnostic & Explainability Suite

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14_App_Router-black?style=flat&logo=next.js&logoColor=white)](https://nextjs.org)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4%2B-38B2AC?style=flat&logo=tailwind-css&logoColor=white)](https://tailwindcss.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**NeuroScan AI** is an assistive diagnostic platform designed to classify brain MRI scans into four distinct intracranial categories (**Glioma**, **Meningioma**, **Pituitary Tumor**, and **Healthy / No Tumor**) while providing **Grad-CAM (Gradient-weighted Class Activation Mapping)** visual heatmaps to explain every prediction.

---

## 🌟 Key Features

- **🔬 Deep Transfer Learning Architecture**: Custom fine-tuned `ResNet-18` backbone optimized specifically for axial brain MRI scans using AdamW, Cosine Annealing, and full-context medical transforms.
- **👁️ Grad-CAM Neural Explainability**: Delineates exact anatomical focus areas on the final convolutional layer (`layer4`), enabling radiologists to inspect the morphological evidence driving each classification.
- **⚡ High-Throughput FastAPI Backend**: Asynchronous REST API supporting multipart form-data image uploads, in-memory base64 image encoding, CORS support, and automatic ephemeral file cleanup.
- **💻 Next.js 14 Medical Dashboard**:
  - Clean, professional healthcare-grade UI styled with Tailwind CSS.
  - Native drag-and-drop MRI upload with real-time client-side preview.
  - Color-coded diagnostic badges (Emerald for Healthy, Rose for Glioma, Amber for Meningioma, Purple for Pituitary).
  - Side-by-side comparative inspection between original MRI scan and Grad-CAM heatmap.
  - Interactive class probability distribution chart powered by `recharts`.
  - One-click client-side clinical summary text report generator.
- **📊 Comprehensive Clinical Metrics Module**: Detailed calculation of True Positives, False Positives, False Positive Rates (FPR), Precision, Recall, and confusion matrix tables.

---

## 📈 Benchmark & Performance

Evaluated across **1,600 independent test scans** in `data/Testing`:

| Class | True Positives (TP) | False Positives (FP) | False Negatives (FN) | False Positive Rate (FPR) | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Glioma** | 304 | **0** | 96 | **0.00%** | **100.00%** | 76.00% |
| **Meningioma** | 384 | 52 | 16 | 4.33% | 88.07% | 96.00% |
| **No Tumor** | 400 | 31 | 0 | 2.58% | 92.81% | **100.00%** |
| **Pituitary** | 400 | 29 | 0 | 2.42% | 93.24% | **100.00%** |

### Clinical Screening Highlights
- **Zero Healthy False Alarms**: `0.00%` False Positive Rate on healthy patients (400 / 400 healthy scans identified with 100% recall).
- **Zero Glioma False Positives**: `100.00%` precision for high-grade glioma detection.
- **Peak Test Accuracy**: **95.56%** (Training Accuracy: **99.91%**).

---

## 🏗️ System Architecture

```text
  [ User Browser / Client ]
             │
      (Drag & Drop MRI)
             ▼
  [ Next.js 14 App Router ] ── (Port 3000)
             │
   POST /api/predict (Multipart Form-Data)
             ▼
  [ FastAPI Microservice ] ── (Port 8000)
             │
             ├──► Preprocessing (224x224, ImageNet Normalization)
             │
             ├──► PyTorch Inference (ResNet-18 Fine-Tuned)
             │        └── Softmax Probabilities & Predicted Class
             │
             ├──► Grad-CAM Activation Engine
             │        └── Target Layer: model.layer4[-1]
             │        └── Heatmap Overlay onto Original Scan
             │
             ▼
  [ JSON Response ]
       ├── predicted_class: "glioma"
       ├── confidence: 0.942
       ├── all_probabilities: { glioma: 0.942, ... }
       └── heatmap_base64: "data:image/jpeg;base64,..."
```

---

## 📁 Repository Structure

```text
NeuroScan-AI/
├── app/                        # Next.js 14 App Router
│   ├── favicon.ico
│   ├── globals.css             # Tailwind CSS definitions
│   ├── layout.tsx              # Root HTML & metadata wrapper
│   └── page.tsx                # Single-page diagnostic dashboard
├── data/                       # MRI Dataset (4 tumor classes)
│   ├── Training/               # glioma, meningioma, notumor, pituitary
│   └── Testing/                # Validation scans (1,600 test cases)
├── evaluate_model.py           # Evaluation script for accuracy, FPR & confusion matrix
├── gradcam_inference.py        # Core inference & Grad-CAM base64 generation
├── main.py                     # FastAPI backend application
├── requirements.txt            # Python dependencies
├── train.py                    # PyTorch high-accuracy training pipeline
├── package.json                # Frontend dependencies
├── tailwind.config.ts          # Tailwind styling configuration
└── tsconfig.json               # TypeScript configuration
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python**: 3.10 or higher
- **Node.js**: 18.x or higher (`npm` / `pnpm` / `yarn`)
- Apple Silicon (MPS) or NVIDIA GPU (CUDA) recommended for accelerated training

---

### 2. Backend Setup (FastAPI & PyTorch)

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI backend server**:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   - API Status: [http://localhost:8000](http://localhost:8000)
   - Interactive Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 3. Frontend Setup (Next.js 14)

1. **Install Node.js packages**:
   ```bash
   npm install
   ```

2. **Start the frontend development server**:
   ```bash
   npm run dev
   ```
   - Open [http://localhost:3000](http://localhost:3000) in your web browser.

---

## 🛠️ CLI Tools & Scripts

### Model Evaluation & False Positive Analysis
Evaluate the model against all test images and output the full statistical breakdown:
```bash
python evaluate_model.py
```

### Standalone Inference & Grad-CAM Heatmap
Run inference on a single image and save the visual output locally:
```bash
python gradcam_inference.py
```

### Re-Train or Fine-Tune the Network
Train ResNet-18 from scratch or fine-tune with custom epochs:
```bash
python train.py
```

---

## 📡 API Reference

### `GET /health`
Verifies backend service health.
```json
{ "status": "ok" }
```

### `POST /api/predict`
Uploads an MRI scan for deep learning classification and explainability.

- **Request**: `multipart/form-data` with key `file` (`.jpg`, `.jpeg`, `.png`)
- **Response**:
```json
{
  "predicted_class": "meningioma",
  "confidence": 0.8554,
  "all_probabilities": {
    "glioma": 0.0034,
    "meningioma": 0.8554,
    "notumor": 0.1403,
    "pituitary": 0.0009
  },
  "heatmap_base64": "/9j/4AAQSkZJRgABAQAAAQ..."
}
```

---

## ⚠️ Clinical Disclaimer

> **IMPORTANT**: NeuroScan AI is developed for academic research, algorithm validation, and clinical decision support. It is **not** a standalone medical diagnostic device. All interpretations must be correlated with clinical history, laboratory findings, and verified by a licensed radiologist or medical specialist.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
