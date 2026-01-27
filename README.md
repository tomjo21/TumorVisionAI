
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

## Run Locally (Easiest Way)
The easiest way to run TumorVision is using Docker, which handles all dependencies and model downloads for you.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

### 1. Build the App
Open your terminal in the project folder and run:
```bash
docker build -t tumorvision .
```

### 2. Run the App
```bash
docker run -p 5000:7860 tumorvision
```

### 3. Access
Open your browser and visit: `http://localhost:5000`

*Note: The first time you run it, it will take ~2 minutes to automatically download the AI models (1.2GB) before the app starts.*

## For Developers (GitHub & Secrets)
If you are cloning this repo and need to configure API keys (e.g., for future AI features):

1.  **Duplicate** `.env.example` and rename it to `.env`.
2.  **Add your secrets** to `.env` (e.g., `API_KEY=...`).
3.  **Run** the app. Docker/Flask will load these variables.

> [!WARNING]
> **Never** commit your `.env` file to GitHub. It is already ignored by `.gitignore`.

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
