import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("model_building.keras")
class_names = list(np.load("processed_data.npz")["class_names"].item().keys())

st.title("Leaf Disease Classification")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (224, 224)) / 255.0

    # Prediction
    pred = model.predict(np.expand_dims(img_resized, axis=0))
    predicted_class = class_names[np.argmax(pred)]

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"**Prediction: {predicted_class}**")
