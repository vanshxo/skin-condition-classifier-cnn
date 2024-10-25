from transformers import AutoImageProcessor, AutoModelForImageClassification
import streamlit as st
import joblib
from PIL import Image
import numpy as np
from PIL import Image
import torch

model_files = {
    'Lung Cancer': 'lung_cancer_model.joblib',
    'Skin Cancer': 'skin_cancer_model.joblib',
    'Breast Cancer': 'breast_cancer_model.joblib',
    'skin condition':'models/model2.joblib'

}

st.title('Disease Prediction System')
st.write('Upload an image to get the disease prediction')

# Dropdown for user to select the disease they want to get predicted
disease_type = st.selectbox(
    'Select the disease you want to get predicted:',
    ['Lung Cancer', 'Skin Cancer', 'Breast Cancer','skin condition','Pneumonia']
)

uploaded_file = st.file_uploader('Upload an Image', type=['jpg', 'png'])


def pn(image):
    processor = AutoImageProcessor.from_pretrained("nickmuchi/vit-finetuned-chest-xray-pneumonia")
    model = AutoModelForImageClassification.from_pretrained("nickmuchi/vit-finetuned-chest-xray-pneumonia")
    # Load the X-ray image
    image = image.convert("RGB")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits and get the predicted class index
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    # Map predicted index to class label
    label = model.config.id2label[predicted_class_idx]
    st.write(f"Predicted class: {label}")


def skin_cancer(img):
    processor = AutoImageProcessor.from_pretrained("nickmuchi/vit-finetuned-chest-xray-pneumonia")
    model = AutoModelForImageClassification.from_pretrained("nickmuchi/vit-finetuned-chest-xray-pneumonia")
    # Load the X-ray image
    image = image.convert("RGB")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits and get the predicted class index
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    prediction_score = torch.softmax(logits, dim=-1)[0][1].item()  

    # Map predicted index to class label
    label = model.config.id2label[predicted_class_idx]
    


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image successfully uploaded.', use_column_width=True)

    if disease_type=='Pneumonia':
        pn(image)







