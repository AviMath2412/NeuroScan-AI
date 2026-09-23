# NeuroScan AI

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14_App_Router-black?style=flat&logo=next.js&logoColor=white)](https://nextjs.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

NeuroScan AI is a deep learning diagnostic assistive platform for intracranial magnetic resonance imaging (MRI) analysis. The system performs multi-class classification across four categories—**Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor (Healthy Control)**—and provides anatomical explainability through Gradient-weighted Class Activation Mapping (Grad-CAM).

The platform includes a PyTorch model training and evaluation pipeline, an asynchronous FastAPI REST backend, a responsive Next.js 14 web dashboard, and containerized Docker Compose orchestration with an Nginx reverse proxy.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Model Performance and Benchmarks](#model-performance-and-benchmarks)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Quick Start: Docker Compose](#quick-start-docker-compose)
- [Manual Setup and Local Development](#manual-setup-and-local-development)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [API Reference](#api-reference)
- [CLI Tools and Pipeline Scripts](#cli-tools-and-pipeline-scripts)
- [Clinical Disclaimer](#clinical-disclaimer)
- [License](#license)

---

## System Architecture

```text
                           [ Client Browser ]
                                   │
                           HTTP Requests (Port 80)
                                   ▼
                      [ Nginx Reverse Proxy ]
                       /                   \
        location /    /                     \   location /api/
                     ▼                       ▼
      [ Next.js Web Frontend ]       [ FastAPI Microservice ]
            (Port 3000)                     (Port 8000)
                                                 │
                                                 ├── Preprocessing (224x224, ImageNet Norm)
                                                 ├── PyTorch ResNet-18 Inference
                                                 └── Grad-CAM Activation Engine
                                                          │
                                                          ▼
                                            JSON Result + In-Memory Base64 Heatmap
```

---

## Model Performance and Benchmarks

The model is based on a transfer-learning `ResNet-18` backbone fine-tuned using AdamW with Cosine Annealing learning rate schedules and label smoothing.

Evaluation performed on **1,600 independent test scans** from `data/Testing`:

### Quantitative Results

| Category | True Positives (TP) | False Positives (FP) | False Negatives (FN) | False Positive Rate (FPR) | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Glioma** | 304 | 0 | 96 | 0.00% | 100.00% | 76.00% |
| **Meningioma** | 384 | 52 | 16 | 4.33% | 88.07% | 96.00% |
| **No Tumor** | 400 | 31 | 0 | 2.58% | 92.81% | 100.00% |
| **Pituitary** | 400 | 29 | 0 | 2.42% | 93.24% | 100.00% |

### Clinical Screening Metrics
- **Healthy Control Recall**: 100.00% (400 / 400 healthy control scans correctly identified).
- **Healthy Screening False Positive Rate**: 0.00% (Zero healthy patients misclassified as harboring a tumor).
- **Glioma Detection Precision**: 100.00% (Zero false positive glioma diagnoses).
- **Peak Validation Accuracy**: 95.56% (Training Accuracy: 99.91%).

---

## Key Features

- **Transfer Learning Backbone**: Fine-tuned ResNet-18 network adapted for axial MRI cranial imaging.
- **Grad-CAM Neural Localization**: Generates pixel-level activation heatmaps from convolutional layer 4 (`layer4[-1]`) to explain regional features influencing classification decisions.
- **Stateless Inference**: Images are processed in-memory or through ephemeral temp storage and returned as base64-encoded visual heatmaps with zero persistent disk leakage.
- **Web Dashboard**: Next.js 14 App Router client with Tailwind CSS, drag-and-drop file ingestion, real-time client-side preview, Recharts probability distributions, and client-side diagnostic report generation.
- **Production Reverse Proxy**: Nginx integration with rate handling, upload size configuration (50MB ceiling), and route rewriting.

---

## Repository Structure

```text
NeuroScan-AI/
├── app/                        # Root Next.js application routes and layouts
├── data/                       # Dataset directory (Training and Testing sets)
│   ├── Training/               # Training partitions (glioma, meningioma, notumor, pituitary)
│   └── Testing/                # Validation partitions (1,600 scans)
├── frontend/                   # Standalone frontend workspace for Docker builds
│   ├── app/                    # Frontend source code
│   ├── Dockerfile              # Multi-stage Next.js standalone container definition
│   └── package.json            # Node.js project manifest
├── docker-compose.yml          # Container configuration for api, web, and nginx
├── Dockerfile                  # FastAPI backend container definition
├── evaluate_model.py           # Evaluation script for accuracy, FPR, and confusion matrix
├── gradcam_inference.py        # Core PyTorch inference and Grad-CAM generation module
├── main.py                     # FastAPI backend application
├── nginx.conf                  # Nginx reverse proxy configuration
├── requirements.txt            # Python dependencies
├── train.py                    # PyTorch model training and fine-tuning pipeline
└── tsconfig.json               # TypeScript configuration
```

---

## Quick Start: Docker Compose

The entire stack (API, Web, and Nginx reverse proxy) can be deployed using Docker Compose:

### 1. Build and Run

```bash
docker compose up -d --build
```

### 2. Verify Container Health

```bash
docker compose ps
```

The output will confirm the active services:
- **`neuroscan-api`**: FastAPI service running on internal port 8000.
- **`neuroscan-web`**: Next.js standalone server running on internal port 3000.
- **`neuroscan-nginx`**: Reverse proxy listening on host port 80.

### 3. Access Endpoints

- **Web Dashboard**: [http://localhost](http://localhost)
- **Direct Backend & Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Direct Frontend**: [http://localhost:3000](http://localhost:3000)

### 4. Stop Services

```bash
docker compose down
```

---

## Manual Setup and Local Development

### Prerequisites

- **Python**: 3.10 or higher
- **Node.js**: 20.x or higher
- **Package Managers**: `pip` and `npm`

---

### Backend Setup

1. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the development server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

---

### Frontend Setup

1. Install Node.js dependencies:
   ```bash
   npm install
   ```

2. Start the Next.js development server:
   ```bash
   npm run dev
   ```

3. Open [http://localhost:3000](http://localhost:3000) in your web browser.

---

## API Reference

### Health Check

`GET /health`

**Response:**
```json
{
  "status": "ok"
}
```

---

### Tumor Classification and Grad-CAM Inference

`POST /api/predict` (also exposed as `POST /predict`)

**Headers:**
`Content-Type: multipart/form-data`

**Body:**
| Field | Type | Description |
| :--- | :--- | :--- |
| `file` | File (`.jpg`, `.jpeg`, `.png`) | Axial brain MRI image |

**Response (`200 OK`):**
```json
{
  "predicted_class": "meningioma",
  "prediction": "Meningioma",
  "confidence": 0.8554,
  "confidence_percentage": "85.54%",
  "all_probabilities": {
    "glioma": 0.0034,
    "meningioma": 0.8554,
    "notumor": 0.1403,
    "pituitary": 0.0009
  },
  "heatmap_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

---

## CLI Tools and Pipeline Scripts

### Full Model Evaluation
Evaluates `best_model.pth` across all 1,600 test images in `data/Testing`:
```bash
python evaluate_model.py
```
Outputs total accuracy, per-class False Positive Rates (FPR), precision, recall, and a formatted confusion matrix.

### Single Image Inference
Runs inference on an individual image and saves the overlaid heatmap locally:
```bash
python gradcam_inference.py
```

### Network Training
Runs the fine-tuning pipeline with Cosine Annealing and data augmentation:
```bash
python train.py
```

---

## Clinical Disclaimer

NeuroScan AI is designed solely for technical evaluation, academic research, and decision-support exploration. It does not constitute a certified medical device and should not be used as a primary diagnostic tool. All clinical interpretations must be validated by a board-certified radiologist or licensed medical professional in correlation with full patient history and laboratory assessments.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for terms and conditions.
