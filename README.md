# 🧘 Yoga-AI: Hybrid Pose Correction & Identification

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose_Tracking-0099CC.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B.svg)

A real-time Computer Vision application that acts as an **AI Yoga Coach**. It features a **Hybrid Architecture** allowing users to switch between:
1.  **Geometry Engine (Rule-Based):** Calculates vector angles for instant form correction (e.g., "Straighten your arm").
2.  **Deep Learning Engine (LSTM):** Uses a trained Neural Network to recognize complex poses and flows from video data.

---

## 🛠️ Prerequisites

You need **Python 3.11+** and **Poetry** installed.

### 1. System Dependencies (Linux/Codespaces)
Since this project processes video, you must install system-level libraries first:

```bash
sudo apt-get update && sudo apt-get install -y \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libavfilter-dev libswscale-dev libswresample-dev \
    pkg-config ffmpeg

----------
pip install poetry
poetry install
----------
poetry run streamlit run app/main.py --server.port 8009

>>> Access the app at http://localhost:8009 (or the Ports tab in Codespaces).

---------
yoga-ai/
└── training_data/
    ├── downdog/       # 5-10 videos
    ├── warrior2/      # 5-10 videos
    └── tree/          # 5-10 videos

---------
poetry run python src/extract_data.py

---------

poetry run python src/train_model.py


📜 License
MIT License. Built for the Production ML Engineering Portfolio.