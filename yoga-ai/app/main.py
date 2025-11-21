import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
import mediapipe as mp
import sys
import os
import collections
import pandas as pd

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import YogaLSTM

st.set_page_config(page_title="Yoga AI Validator", layout="wide")

# --- CONFIG ---
TEST_VIDEO_FOLDER = "test_data"
os.makedirs(TEST_VIDEO_FOLDER, exist_ok=True)

# --- LOAD AI MODEL ---
@st.cache_resource
def load_model():
    if not os.path.exists("model_data/labels.npy"):
        return None, None
    labels = np.load("model_data/labels.npy")
    
    model = YogaLSTM(num_classes=len(labels))
    try:
        model.load_state_dict(torch.load("models/yoga_lstm.pth", map_location=torch.device('cpu')))
        model.eval()
        return model, labels
    except:
        return None, None

model, labels = load_model()

# --- HELPER: EXTRACT KEYPOINTS ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

def extract_keypoints(results):
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        return list(np.array([[res.x, res.y, res.z, res.visibility] for res in lm]).flatten())
    return [0] * 132

# --- UI ---
st.title("🧘 Yoga AI: Validation Lab")

if model is None:
    st.error("❌ Model not found! Please run `src/train_model.py` first.")
    st.stop()

# --- INPUT METHOD SELECTION ---
input_method = st.radio("Choose Input Method:", ["Select from Codespace (Best for Large Files)", "Upload via Browser"])

video_path = None

if input_method == "Upload via Browser":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

elif input_method == "Select from Codespace (Best for Large Files)":
    # List files in the test_videos folder
    files = [f for f in os.listdir(TEST_VIDEO_FOLDER) if f.endswith(('.mp4', '.mov', '.avi'))]
    
    if not files:
        st.warning(f"No videos found in `{TEST_VIDEO_FOLDER}`. Drag and drop your 31MB file into that folder!")
    else:
        selected_file = st.selectbox("Select a Video:", files)
        video_path = os.path.join(TEST_VIDEO_FOLDER, selected_file)

# --- PROCESS VIDEO ---
if video_path:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.video(video_path)

    with col2:
        st.subheader("🔍 AI Analysis")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames_sequence = []
        predictions = []
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % 2 == 0: # Speed up by skipping every other frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                keypoints = extract_keypoints(results)
                
                frames_sequence.append(keypoints)
                
                if len(frames_sequence) == 30:
                    with torch.no_grad():
                        input_tensor = torch.tensor([frames_sequence], dtype=torch.float32)
                        out = model(input_tensor)
                        prob = torch.softmax(out, dim=1)[0]
                        pred_idx = torch.argmax(prob).item()
                        predictions.append(labels[pred_idx])
                    frames_sequence = frames_sequence[5:]
            
            frame_idx += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))
            status_text.text(f"Processing frame {frame_idx}...")

        cap.release()
        progress_bar.empty()
        status_text.empty()

        if predictions:
            counts = collections.Counter(predictions)
            winner, count = counts.most_common(1)[0]
            confidence = (count / len(predictions)) * 100
            
            st.success(f"### Result: {winner}")
            st.metric("Confidence", f"{confidence:.1f}%")
            
            chart_data = pd.DataFrame.from_dict(counts, orient='index', columns=['Votes'])
            st.bar_chart(chart_data)
        else:
            st.warning("Could not detect pose (video might be too short or person not visible).")