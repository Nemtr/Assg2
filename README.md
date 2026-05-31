# 📹 ML-based Encoding Optimizer

## 1. Project Overview

This project, **ML-based Encoding Optimizer (Project Code: 2502k)**, is an intelligent system designed to maximize video compression efficiency. Instead of applying a rigid "one-size-fits-all" encoding configuration, the system automatically analyzes the physical complexity of the input video and calculates the mathematical "sweet spot" for its compression parameters.

The system extracts **Spatial Information (SI)** and **Temporal Information (TI)** metrics using computer vision pipelines, predicts optimal **Quantization Parameters (QP/CRF)** and **Group of Pictures (GOP)** sizes via a **Random Forest Regressor**, and executes hardware-level encoding using **FFmpeg** wrapped inside an interactive **Streamlit** dashboard.

---

## 2. Key Features

AI-Driven Parameter Prediction: Analyzes the physical characteristics of incoming `.mp4` videos using spatial edge density (Sobel Filter) and inter-frame motion vectors.
    
Dynamic Codec Tuning:

    Low-Motion / Static Content: Lengthens GOP structure (up to 90) and raises CRF thresholds to achieve massive storage reduction (up to 95% bandwidth savings).
    
    Perception-Aware Encoding (CRF): Implements Constant Rate Factor coding (`libx264`) rather than standard rigid QP, leveraging human visual boundaries to drop bitrates.

Interactive Web UI: Real-time metrics visualization representing raw features, predicted parameters, and cross-comparison compression ratios.

---

## 3. System Requirements

Operating System: Ubuntu / Linux (highly recommended) or Windows 10/11.

Python Version: `3.10` or `3.11`

External Core Tool: **FFmpeg** and **FFprobe** must be installed and properly configured in the system `PATH`.

---

## 4. Installation & Setup Instructions

### Step 1: Install System Dependencies

Open your operating system terminal and install Python environment tools alongside the FFmpeg multimedia framework:

sudo apt update

sudo apt install python3 python3-pip python3-venv ffmpeg -y

(Windows users should download the FFmpeg essentials build from the official site and add its \bin directory to the system environment variables)

### Step 2: Navigate to Project Workspace

cd assignment2

### Step 3: Instantiate and Activate Virtual Environment

python3 -m venv .venv

source .venv/bin/activate

### Step 4: Install Python Packages

pip install --upgrade pip

pip install -r requirements.txt

(Note: The environment depends on streamlit, opencv-python, numpy, pandas, scikit-learn, and joblib)

## 5. How to run - using streamlit 

Ensure your virtual environment (.venv) is active, then launch the main interface application:

streamlit run src/app.py

## 6. Project structure
assignment2/

├── data/

│   ├── raw/                  # Source .mp4 database for metric extraction

│   ├── temp/                 # Caches original uploads and optimized outputs

│   └── video_dataset.csv     # Extracted features and heuristic labeled dataset

├── models/

│   ├── model_gop.pkl         # Trained Random Forest Regressor for GOP bounds

│   └── model_qp.pkl          # Trained Random Forest Regressor for CRF steps

├── src/

    ├── app.py                # Multi-column interactive UI dashboard logic
    
    ├── features.py           # OpenCV Sobel matrix pipelines and metadata extraction
    
    ├── train_ml.py           # Ensemble SciKit-Learn mapping & serialization core
    
    └── video_coder.py        # FFmpeg dynamic assembly core & guardrail execution
    
├── README.md

└── requirements.txt          # Python application package dependencies

## 7. Authors

Trần Lê Hải Nam (20213580) - Lead Developer for System Design & Backend.

Nguyễn Hữu Mạnh (20224288) - Lead Developer for Machine Learning & Evaluation.

Supervisor: TS. Phạm Văn Tiến

Project management link (Trello): https://trello.com/b/2f1nnEKv/assg2

Link to the dataset (data/raw): https://drive.google.com/drive/folders/1iMnPJBqEXA7cul_QOurXc_Tz8WVm-79b?usp=sharing


