import streamlit as st
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch

# Load pre-trained model and image processor
@st.cache_resource
def load_model():
    model_name = "timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
    model = AutoModelForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor

model, processor = load_model()

# Streamlit App
st.title("Image Classification with Hugging Face")
st.write("Upload an image to classify it using a pre-trained model.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    
    # Classify image
    st.write("Classifying...")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    # Get label
    label = model.config.id2label[predicted_class_idx]
    st.write(f"Predicted Class: **{label}**")
