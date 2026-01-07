import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import requests

st.set_page_config(page_title="AI Crack Detection App", layout="centered")
st.title("🧠 Concrete Crack Detection")
st.write("Upload an image to detect concrete cracks using AI.")

MODEL_ID = "1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"
MODEL_PATH = "crack_model.h5"

@st.cache_resource
def load_crack_model():

    # Download only the first time
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model from Google Drive..."):
            r = requests.get(MODEL_URL, allow_redirects=True)
            open(MODEL_PATH, "wb").write(r.content)
            st.success("📦 Model downloaded successfully!")

    return tf.keras.models.load_model(MODEL_PATH)

model = load_crack_model()

uploaded_file = st.file_uploader(
    "📷 Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    st.write(f"🔍 Confidence Score: `{prediction:.2f}`")

    if prediction > 0.5:
        st.error("⚠️ Crack Detected!")
    else:
        st.success("✅ No Crack Detected!")
