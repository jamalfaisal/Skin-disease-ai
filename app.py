import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import h5py

import os
import gdown

MODEL_URL = "https://drive.google.com/uc?id=1Zwggbh4accCiIS0TstekUinKGTn1wIWZ"

if not os.path.exists("skin_model.h5"):
    gdown.download(MODEL_URL, "skin_model.h5", quiet=False)

# Load model
model = tf.keras.models.load_model("skin_model.h5")

# Load class names
h5f = h5py.File("processed_data/dataset.h5", "r")
class_names = [name.decode("utf-8") for name in h5f["class_names"][:]]

st.title("🧠 Skin Disease Detection AI")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224,224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    x = np.array(img)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)
    class_index = np.argmax(pred)
    confidence = np.max(pred)

    st.success(f"Prediction: {class_names[class_index]}")
    st.info(f"Confidence: {confidence:.2f}")