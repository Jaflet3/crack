import streamlit as st
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

st.title("Wall vs Road Crack Detection 🔍")
st.write("Upload a cracked wall or road image to see crack highlights and a predicted type.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=['jpg','png','jpeg'])

# Mock prediction function
def mock_predict(img):
    # For demonstration: randomly predict wall or road
    # Replace this with actual model prediction later
    return random.choice(['walls', 'road'])

# Crack detection & visualization
def detect_crack(img):
    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Predict type (mock)
    label = mock_predict(img)

    # Plot original and edge images side by side
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(edges, cmap='gray')
    ax[1].set_title("Crack Highlighted")
    ax[1].axis('off')

    st.pyplot(fig)
    st.markdown(f"### Predicted: **{label}**")

# Run detection on uploaded image
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    detect_crack(img)
