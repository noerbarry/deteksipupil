import cv2
import streamlit as st
import numpy as np

# Try accessing the camera by name
camera_name = '/dev/video0'  # Replace with the name or identifier of your camera device
video_capture = cv2.VideoCapture(camera_name, cv2.CAP_V4L2)

def detect_sexual_arousal(frame):
    # Face detection using Haar Cascade Classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # If no faces are detected, return an empty list
    if len(faces) == 0:
        return []
    
    # If faces are detected, return the detected face locations
    return faces

# Web interface using Streamlit
st.title("Pupil Dilation Detection")

# Add a "Turn on Camera" checkbox
camera_on = st.checkbox("Turn on Camera")

# Use the camera for video streaming
video_capture = None
frame_container = st.empty()

while camera_on:
    if video_capture is None:
        video_capture = cv2.VideoCapture(0)

    if video_capture is not None and video_capture.isOpened():
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        
        # Detect pupil dilation (sexual arousal)
        faces = detect_sexual_arousal(frame)
        
        # Display the detection result
        if len(faces) > 0:
            st.write("High pupil dilation detected. (Possible sexual arousal)")
        else:
            st.write("Normal pupil dilation. (No sexual arousal detected)")
        
        # Display the frame with rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Convert the BGR image format to RGB before displaying it in Streamlit
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame in Streamlit
        frame_container.image(rgb_frame)

# Turn off the camera when the checkbox is inactive (unchecked)
if video_capture is not None:
    video_capture.release()
    cv2.destroyAllWindows()

