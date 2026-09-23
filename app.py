import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
from datetime import datetime, timedelta
from ultralytics import YOLO

st.set_page_config(page_title="CR Headcount", page_icon="📸", layout="centered")

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

st.title("📸 CR's Crowd Counter (YOLOv8)")

with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider("Detection Sensitivity", 0.10, 0.90, 0.25, 0.05)
    
    display_mode = st.radio(
        "Marker Style:",
        ["Green Dots", "Numbered Dots", "Clean Boxes (No Text)"]
    )
    
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
    
    # Run YOLOv8 detection
    results = model.predict(source=image, conf=confidence, classes=[0])
    boxes = results[0].boxes
    detected_count = len(boxes)

    # Base image for drawing
    img_display = np.array(image).copy()

    if display_mode == "Clean Boxes (No Text)":
        # Draw clean boxes without labels or confidence scores
        annotated_bgr = results[0].plot(labels=False, conf=False)
        img_display = annotated_bgr[..., ::-1]  # Convert BGR to RGB
    else:
        # Draw Green Dots or Numbered Dots
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Place the dot near the top center of the person's box (head level)
            cx = int((x1 + x2) / 2)
            cy = int(y1 + (y2 - y1) * 0.18)

            # Outer dark ring + vibrant neon green center for high contrast
            cv2.circle(img_display, (cx, cy), 7, (0, 0, 0), -1)
            cv2.circle(img_display, (cx, cy), 5, (0, 255, 64), -1)

            if display_mode == "Numbered Dots":
                cv2.putText(
                    img_display,
                    str(idx + 1),
                    (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    img_display,
                    str(idx + 1),
                    (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 64),
                    1
                )

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
    st.image(img_display, caption=f"Detection Map ({display_mode})", use_container_width=True)

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
