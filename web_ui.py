import streamlit as st
import cv2
from video_detector import VideoDetector

st.title("🛡️ Deepfake Detector Web UI")
uploaded_file = st.file_uploader("Upload MP4/AVI/MOV", type=['mp4','avi','mov'])

if uploaded_file:
    # Save uploaded file
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.video(uploaded_file)
    
    if st.button("🔍 ANALYZE DEEPFAKE"):
        detector = VideoDetector()
        result = detector.analyze_video("temp_video.mp4", show_preview=False)
        st.success(f"**{result[1]}** ({result[0]:.3f})")