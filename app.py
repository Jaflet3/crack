# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import gdown
from tensorflow.keras.models import load_model
from fpdf import FPDF

# -------------------------
# Streamlit UI setup
# -------------------------
st.set_page_config(page_title="🛠️ Crack Detection", layout="centered")
st.title("🛠️ Image-based Crack Detection")
st.markdown(
    "Upload an image to detect cracks, calculate severity, and download a report."
)

# -------------------------
# 1️⃣ Download model from Google Drive
# -------------------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the trained model
model = load_model(MODEL_PATH)

# -------------------------
# 2️⃣ Functions
# -------------------------
def calculate_crack_severity(image_path):
    """Calculate crack severity as % of image area covered by cracks."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    crack_pixels = np.sum(thresh == 255)
    total_pixels = thresh.size
    severity_score = (crack_pixels / total_pixels) * 100
    return round(severity_score, 2)


def create_pdf_report(image_path, severity_score, report_path="report.pdf"):
    """Generate a PDF report with the image and severity score."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Crack Detection Report", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Severity Score: {severity_score}%", ln=True)
    pdf.ln(10)
    pdf.image(image_path, x=50, w=100)
    pdf.output(report_path)
    return report_path


def predict_crack(image_path):
    """Use the trained model to predict crack presence."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224)
