import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from datetime import datetime, timedelta
from ultralytics import YOLO

st.set_page_config(page_title="CR Headcount", page_icon="📸", layout="centered")

# Cache the YOLO model so it only loads once and keeps your app fast
@st.cache_resource
def load_model():
    # YOLOv8 nano model is lightweight and auto-downloads the first time it runs
    return YOLO('yolov8n.pt')

model = load_model()

st.title("📸 CR's Crowd Counter (YOLOv8)")
st.info("For 90+ students, take a high-resolution photo with your native phone camera and use the Upload button instead of the live camera input.")

with st.sidebar:
    st.header("⚙️ Settings")
    # YOLO confidence is usually higher than MediaPipe; 0.25 is a good baseline
    confidence = st.slider("Detection Sensitivity", 0.10, 0.90, 0.25, 0.05)
    
    if st.button("Reset Manual Adjustments"):
        st.session_state.manual_offset = 0
        st.rerun()

if "manual_offset" not in st.session_state:
    st.session_state.manual_offset = 0

photo = st.camera_input("Take a picture")
uploaded_file = st.file_uploader("Upload high-res classroom photo", type=["jpg", "jpeg", "png"])

image_source = uploaded_file or photo

if image_source:
    image = Image.open(image_source)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    
    # Run YOLOv8 detection. classes=[0] restricts it to ONLY detect 'persons'
    results = model.predict(source=image, conf=confidence, classes=[0])
    
    # Extract the total count of detected persons
    detected_count = len(results[0].boxes)
    
    # Generate the image with bounding boxes drawn
    img_with_boxes = results[0].plot()

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

    st.caption(f"YOLO detected: {detected_count} | Manual adjustment: {st.session_state.manual_offset:+d}")
    
    # Show the bounding box image (YOLO outputs BGR, so convert to RGB for Streamlit)
    st.image(img_with_boxes[..., ::-1], caption="YOLO Detection Map", use_container_width=True)

    ist_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
    time_str = ist_time.strftime("%Y-%m-%d %I:%M %p")
    
    report_text = f"Class Headcount Report\nDate & Time: {time_str}\nTotal Present: {final_count}"
    
    st.download_button(
        label="📥 Download Report",
        data=report_text,
        file_name=f"Attendance_{ist_time.strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True
    )
