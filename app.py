import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import gdown

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI Crack Detection App",
    layout="centered"
)

st.title("🧠 Concrete Crack Detection")

# ---------------- MODEL CONFIG ---------------- #
MODEL_FILE_ID = "1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"
MODEL_PATH = "crack_model.h5"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"

# ---------------- DOWNLOAD MODEL ---------------- #
@st.cache_resource
def load_crack_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model from Google Drive..."):
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
            st.success("📦 Model downloaded successfully!")

    return tf.keras.models.load_model(MODEL_PATH)

model = load_crack_model()

# ---------------- FILE UPLOAD ---------------- #
uploaded_file = st.file_uploader(
    "Upload a concrete surface image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load & preprocess image
    img = load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    st.write(f"🔍 Prediction confidence: **{prediction:.2f}**")

    if prediction > 0.5:
        st.error("⚠ Crack Detected!")
    else:
        st.success("✔ No Crack Detected!")
