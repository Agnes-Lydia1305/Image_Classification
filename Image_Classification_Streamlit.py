import streamlit as st
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch

# Load pre-trained model and feature extractor
@st.cache_resource
def load_model():
    model_name = "google/vit-base-patch16-224"
    model = ViTForImageClassification.from_pretrained(model_name)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    return model, feature_extractor

model, feature_extractor = load_model()

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
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Classify image
    st.write("Classifying...")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    # Get label
    label = model.config.id2label[predicted_class_idx]
    st.write(f"Predicted Class: **{label}**")
