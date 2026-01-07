import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import gdown

st.set_page_config(page_title="AI Crack Detection", layout="centered")
st.title("🧠 Concrete Crack Detection")
st.write("Upload an image to detect cracks using AI.")

# -------- MODEL FILE (weights) -------- #
MODEL_ID = "1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"   # <-- use your file id ONLY if using Drive
WEIGHTS_PATH = "crack_weights.weights.h5"

# -------- DOWNLOAD WEIGHTS IF NOT FOUND -------- #
if not os.path.exists(WEIGHTS_PATH):
    with st.spinner("⬇️ Downloading weights..."):
        gdown.download(id=MODEL_ID, output=WEIGHTS_PATH, quiet=False)
        st.success("Weights downloaded!")

# -------- REBUILD MODEL ARCHITECTURE -------- #
@st.cache_resource
def build_model():

    model = models.Sequential([
        layers.Input(shape=(150,150,3)),

        layers.Conv2D(32, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.load_weights(WEIGHTS_PATH)
    return model


model = build_model()

# -------- UPLOAD IMAGE -------- #
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = load_img(uploaded_file, target_size=(150,150))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    st.write(f"🔍 Confidence: `{pred:.2f}`")

    if pred > 0.5:
        st.error("⚠️ Crack Detected")
    else:
        st.success("✅ No Crack Detected")
