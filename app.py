import streamlit as st
import numpy as np
import mediapipe as mp
from PIL import Image

st.set_page_config(page_title="CR Headcount", page_icon="📸", layout="centered")

st.title("📸 CR's Headcount App")

# Sensitivity controls in an expander / sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider("Detection Sensitivity (lower = detects more faces)", 0.10, 0.90, 0.25, 0.05)

# Session state for manual correction
if "manual_offset" not in st.session_state:
    st.session_state.manual_offset = 0

photo = st.camera_input("Take a picture of the class")
uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

image_source = photo or uploaded_file

if image_source:
    image = Image.open(image_source)
    img_array = np.array(image)

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=confidence) as face_detection:
        results = face_detection.process(img_array)
        detected_count = len(results.detections) if results.detections else 0

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(img_array, detection)

    final_count = max(0, detected_count + st.session_state.manual_offset)

    st.divider()
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.metric(label="🎓 Final Headcount", value=final_count)
    with col2:
        if st.button("➕ Add 1"):
            st.session_state.manual_offset += 1
            st.rerun()
    with col3:
        if st.button("➖ Sub 1"):
            st.session_state.manual_offset -= 1
            st.rerun()

    st.caption(f"AI detected: {detected_count} | Manual adjustment: {st.session_state.manual_offset:+d}")
    st.image(img_array, caption="Detection Map", use_container_width=True)
