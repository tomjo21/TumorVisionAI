---
title: TumorVision AI
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# TumorVision - AI-Powered Brain Tumor Analysis Platform

A Flask-based web application for automated brain tumor detection, classification, and 3D visualization using deep learning.

## Features

- **2D Classification**: Multi-class tumor detection (Glioma, Meningioma, Pituitary, No Tumor)
- **3D Segmentation**: Voxel-level tumor mapping with sub-region identification
- **Interactive 3D Visualization**: WebGL-based tumor model viewer
- **Progress Tracking**: Longitudinal analysis for tumor growth monitoring
- **Medical Reports**: Automated PDF report generation
- **AI Chatbot**: Olmo assistant for medical imaging guidance

## Tech Stack

- **Backend**: Flask, TensorFlow/Keras
- **Frontend**: Vanilla JavaScript, Three.js
- **ML Models**: Ensemble 2D CNN, 3D U-Net
- **Medical Imaging**: NiBabel, NumPy

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd MainProject

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The application will be available at `http://localhost:5000`

## Project Structure

```
MainProject/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── models/               # Pre-trained ML models
├── static/               # CSS, JavaScript, uploads
├── templates/            # HTML templates
├── utils.py              # Preprocessing & visualization
├── utils_mesh.py         # 3D mesh generation
└── utils_report.py       # PDF report generation
```

## Usage

1. Navigate to the homepage
2. Choose analysis type (2D Classification or 3D Segmentation)
3. Upload MRI scan (JPG/PNG for 2D, NIfTI for 3D)
4. View results and download medical report

## Models

- **2D Classifier**: Ensemble model trained on brain MRI dataset
- **3D Segmenter**: U-Net architecture for multi-region tumor segmentation

## License

Educational/Research Use
