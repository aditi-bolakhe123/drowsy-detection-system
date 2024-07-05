import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import tempfile

# Load the trained model
model = tf.keras.models.load_model("C:/Users/Aditi Bolakhe/Desktop/drowsy detection/DDD/working code/densenet.h5")

# Define a function to preprocess the uploaded image/frame
def preprocess_image(image):
    img = cv2.resize(image, (224, 224))  # Resize the image to the input size required by the model
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Define a function to predict drowsiness
def predict(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    return prediction

# Streamlit app
st.title("Drowsy Driver Detection System")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Home", "Image Classification", "Video Classification", "Webcam Classification", "About"])

if options == "Home":
    st.write("""
    ## Welcome to the Drowsy Driver Detection System
    Use the sidebar to navigate to different sections of the application.
    - **Image Classification**: Upload an image to detect drowsiness.
    - **Video Classification**: Upload a video to detect drowsiness frame by frame.
    - **Webcam Classification**: Use your webcam to detect drowsiness in real-time.
    """)

elif options == "Image Classification":
    st.header("Image Classification")
    uploaded_image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image_file is not None:
        image = Image.open(uploaded_image_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")
        
        prediction = predict(np.array(image))
        if prediction[0][0] < 0.5:
            st.error("The person is drowsy.")
        else:
            st.success("The person is not drowsy.")

elif options == "Video Classification":
    st.header("Video Classification")
    uploaded_video_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
    if uploaded_video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video_file.read())
        
        vf = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break

            # Predict drowsiness for the current frame
            prediction = predict(frame)
            if prediction[0][0] < 0.5:
                label = "Drowsy"
                color = (0, 0, 255)  # Red for drowsy
            else:
                label = "Active"
                color = (0, 255, 0)  # Green for active

            # Annotate the frame with the prediction
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            # Display the annotated frame
            stframe.image(frame, channels="BGR")
        
        vf.release()

elif options == "Webcam Classification":
    st.header("Webcam Classification")
    run_webcam = st.checkbox("Run Webcam")
    if run_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture image")
                break
            
            # Convert color from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Predict drowsiness
            prediction = predict(frame)
            if prediction[0][0] < 0.5:
                label = "Drowsy"
                color = (255, 0, 0)  # Red for drowsy
            else:
                label = "Active"
                color = (0, 255, 0)  # Green for active

            # Annotate the frame with the prediction
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Display the frame
            stframe.image(frame, channels="RGB")

            # Check if the user wants to stop the webcam
            run_webcam = st.checkbox("Run Webcam", value=True)

        cap.release()

elif options == "About":
    st.header("About")
    st.markdown("""
    This application uses a pre-trained model to detect drowsiness from images, videos, and live webcam feed.
    Upload an image or video file or toggle the 'Run Webcam' checkbox to use the webcam. The model predicts whether
    the person in the media is drowsy or active.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This application uses a pre-trained model to detect drowsiness from images, videos, and live webcam feed.
Upload an image or video file or toggle the 'Run Webcam' checkbox to use the webcam. The model predicts whether
the person in the media is drowsy or active.
""")
