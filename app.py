import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

st.set_page_config(page_title="CR Headcount", page_icon="📸")

st.title("📸 CR's Headcount App")
st.write("Take a photo of the class or upload one to get an instant headcount!")

# Mobile-friendly inputs
photo = st.camera_input("Take a picture of the class")
uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

image_source = photo or uploaded_file

if image_source:
    # Convert the uploaded file to an OpenCV image
    image = Image.open(image_source)
    img_array = np.array(image)
    
    # Initialize MediaPipe Face Detection (model_selection=1 is better for faces further away)
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        # Process the image
        results = face_detection.process(img_array)
        
        count = 0
        if results.detections:
            count = len(results.detections)
            # Draw boxes around faces so you can verify who was counted
            for detection in results.detections:
                mp_drawing.draw_detection(img_array, detection)
        
        # Big metric display for the count
        st.metric(label="🎓 Total Students Detected", value=count)
        
        # Show the processed image
        st.image(img_array, caption="Verified Headcount", use_container_width=True)
