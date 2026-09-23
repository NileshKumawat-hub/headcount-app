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
    
    dot_scale = st.slider("Marker Size Multiplier", 0.5, 3.0, 1.2, 0.1)
    
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
    
    results = model.predict(source=image, conf=confidence, classes=[0])
    boxes = results[0].boxes
    detected_count = len(boxes)

    img_display = np.array(image).copy()
    h, w, _ = img_display.shape

    # Calculate proportional scale based on image dimensions
    base_dim = max(h, w)
    res_factor = base_dim / 1000.0 * dot_scale

    if display_mode == "Clean Boxes (No Text)":
        annotated_bgr = results[0].plot(labels=False, conf=False, line_width=max(2, int(2 * res_factor)))
        img_display = annotated_bgr[..., ::-1]
    else:
        radius_inner = max(4, int(7 * res_factor))
        radius_outer = max(6, int(10 * res_factor))
        font_scale = max(0.4, 0.5 * res_factor)
        thickness_text = max(1, int(1.5 * res_factor))

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Position dot at head/upper-torso level
            cx = int((x1 + x2) / 2)
            cy = int(y1 + (y2 - y1) * 0.18)

            # High-visibility marker: dark outer border + vivid neon green core
            cv2.circle(img_display, (cx, cy), radius_outer, (0, 0, 0), -1)
            cv2.circle(img_display, (cx, cy), radius_inner, (0, 255, 64), -1)

            if display_mode == "Numbered Dots":
                num_str = str(idx + 1)
                offset_x = cx + int(12 * res_factor)
                offset_y = cy - int(4 * res_factor)
                
                # Dark outline for text contrast
                cv2.putText(
                    img_display,
                    num_str,
                    (offset_x, offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    thickness_text + 2
                )
                # White fill for legibility
                cv2.putText(
                    img_display,
                    num_str,
                    (offset_x, offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness_text
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
