import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model('cnn_model.h5')

# Define class names in the same order as training
class_names = ['cats', 'dogs', 'snakes']

# Streamlit UI
st.title("🐾 Animal Classifier - Cats, Dogs, Snakes")
st.write("Upload an image and I’ll tell you what animal it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")

    # Preprocess the image
    img = img.resize((256, 256))  # Match model input size
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Display prediction
    st.write(f"### 🧠 Prediction: **{class_names[predicted_class]}**")
    st.write(f"Confidence: `{confidence * 100:.2f}%`")
