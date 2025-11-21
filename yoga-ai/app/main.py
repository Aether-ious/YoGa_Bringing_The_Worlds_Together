import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import sys
import os

# Path Fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pose_engine import YogaPoseAnalyzer

st.set_page_config(page_title="Hybrid Yoga AI", layout="wide")
st.title("🧘 Hybrid Yoga Trainer")

# --- SIDEBAR CONTROLS ---
st.sidebar.title("⚙️ Settings")
mode_selection = st.sidebar.radio(
    "Select Intelligence Mode:",
    ("Geometry (Rule-Based)", "Deep Learning (LSTM)")
)

# Map selection to code key
mode_key = "geometry" if "Geometry" in mode_selection else "ai"

st.sidebar.info(
    """
    **Geometry Mode:** Uses math rules. Fast, precise feedback. Good for alignment.
    
    **AI Mode:** Uses trained Neural Network. Recognizes flows and complex shapes. Requires training data.
    """
)

# Initialize Logic based on selection
analyzer = YogaPoseAnalyzer(mode=mode_key)

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # The analyzer state persists as long as the stream runs
        processed_img = analyzer.process_frame(img)
        
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# Start Stream
webrtc_streamer(
    key="yoga-hybrid",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)