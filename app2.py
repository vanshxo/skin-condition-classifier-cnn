import joblib
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import streamlit as st
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


def lung(image):
    processor = AutoImageProcessor.from_pretrained("ebmonser/lung-cancer-image-classification")
    model = AutoModelForImageClassification.from_pretrained("ebmonser/lung-cancer-image-classification")
    image = image.convert("RGB")
    img = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**img)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    # Map predicted index to class label
    # The model should have a config with the class labels
    label = model.config.id2label[predicted_class_idx]
    
    # Extract prediction
    
    prediction_score = torch.softmax(logits, dim=-1)
    confidence = prediction_score.max()
    
    # Update prediction data
    prediction_data = {
        'result': label,
        'confidence': confidence
        
    }
    return prediction_data,model.config.id2label,prediction_score
# Inference code for Pneumonia
def predict_pneumonia(image, content):
    processor = AutoImageProcessor.from_pretrained("nickmuchi/vit-finetuned-chest-xray-pneumonia")
    model = AutoModelForImageClassification.from_pretrained("nickmuchi/vit-finetuned-chest-xray-pneumonia")
    # Preprocess the image for the model
    image = image.convert("RGB")
    img = processor(images=image, return_tensors="pt")
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**img)
    
    # Extract prediction
    logits = outputs.logits
    prediction_score = torch.softmax(logits, dim=-1)[0][1].item()  # Assuming index 1 is "Pneumonia" in id2label
    
    # Plot confidence gauge
    st.plotly_chart(create_confidence_gauge(prediction_score, 'Pneumonia Detection Confidence'))
    
    # Determine result based on confidence threshold
    
    result = content["pneumonia_result"].format(prediction=round(prediction_score * 100, 2))
    confidence = prediction_score

# Inference code for Pneumonia
def predict_skin_cancer(image, content):
    processor = AutoImageProcessor.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")
    model = AutoModelForImageClassification.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")
    # Preprocess the image for the model
    image = image.convert("RGB")
    img = processor(images=image, return_tensors="pt")
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**img)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    # Map predicted index to class label
    # The model should have a config with the class labels
    label = model.config.id2label[predicted_class_idx]
    
    # Extract prediction
    
    prediction_score = torch.softmax(logits, dim=-1)
    confidence = prediction_score.max()
    
    # Update prediction data
    prediction_data = {
        'result': label,
        'confidence': confidence
        
    }
    return prediction_data,model.config.id2label,prediction_score
# Initialize session state variables
if 'language' not in st.session_state:
    st.session_state.language = "English"
if 'history' not in st.session_state:
    st.session_state.history = []

def save_prediction_history(prediction_data):
    """Save prediction to session state history"""
    prediction_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append(prediction_data)

def get_confidence_color(confidence):
    """Return color based on confidence level"""
    if confidence >= 0.8:
        return "green"
    elif confidence >= 0.6:
        return "orange"
    return "red"

def preprocess_image(image, target_size, normalize=False):
    """Preprocess image consistently for all models"""
    img = image.resize(target_size)
    img = np.array(img)
    if len(img.shape) == 2:
        img = np.stack((img,)*3, axis=-1)
    img = np.expand_dims(img, axis=0)
    if normalize:
        img = img / 255.0
    return img

def create_confidence_gauge(confidence, title):
    """Create a gauge chart for confidence visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': get_confidence_color(confidence)},
            'steps': [
                {'range': [0, 60], 'color': "lightgray"},
                {'range': [60, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "darkgray"}
            ],
        }
    ))
    fig.update_layout(height=250)
    return fig

def predict_disease(image, disease_type, model_files, content):
    """Enhanced unified prediction function for all disease types"""
    try:
        prediction_data = {
            'disease_type': disease_type,
            'language': st.session_state.language
        }
        
        if disease_type in ['Tuberculosis', 'तपेदिक']:
            img = preprocess_image(image, (320, 320), normalize=True)
            with st.spinner(content["loading_model"]):
                model = joblib.load(model_files[disease_type])
            prediction = model.predict(img)[0]
            
            # Create confidence visualization
            st.plotly_chart(create_confidence_gauge(prediction, 'Tuberculosis Detection Confidence'))
            
            if prediction > 0.975:
                result = content["tuberculosis_result"].format(prediction=round(prediction*100, 2))
                confidence = prediction
            else:
                result = content["tuberculosis_normal"]
                confidence = 1 - prediction
                
            prediction_data.update({
                'result': result,
                'confidence': confidence
            })
            
            # Display result in a nice card
            with st.container():
                st.markdown(f"""
                <div style='background-color: rgba(255, 255, 255, 0.7); padding: 20px; border-radius: 10px;'>
                    <h3 style='color: {get_confidence_color(confidence)};'>{result}</h3>
                    <p>Confidence: {confidence*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
        elif disease_type in ['Skin Condition', 'त्वचा की स्थिति']:
            img = preprocess_image(image, (224, 224), normalize=False)
            with st.spinner(content["loading_model"]):
                model = joblib.load(model_files[disease_type])
            prediction = model.predict(img)
            confidence = np.max(prediction)
            class_idx = np.argmax(prediction)
            
            predictions = {
                0: 'Carcinoma', 
                1: 'Keratosis',
                2: 'Acne',
                3: 'Eczema',
                4: 'Rosacea',
                5: 'Milia'
            }
            with st.container():
                st.markdown(f"""
                <div style='background-color: rgba(255, 255, 255, 0.7); padding: 20px; border-radius: 10px;'>
                    <h3 style='color: {get_confidence_color(confidence)};'>
                        {content['detected_condition']}: {result}
                    </h3>
                    <p>Confidence: {confidence*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            probs_df = pd.DataFrame({
                'Condition': predictions.values(),
                'Probability': prediction[0] * 100
            })
            fig = px.bar(probs_df, x='Condition', y='Probability',
                        title='Probability Distribution of Conditions')
            st.plotly_chart(fig)
        
            
        
        elif disease_type in ['Pneumonia', 'न्यूमोनिया']:
            # Use the predict_pneumonia function to get the result and confidence
            prediction_data = predict_pneumonia(image, content)
            confidence = prediction_data['confidence']
            result = prediction_data['result']
            

                        
            # Display result with a modern card design
            with st.container():
                st.markdown(f"""
                <div style='background-color: rgba(255, 255, 255, 0.7); padding: 20px; border-radius: 10px;'>
                    <h3 style='color: {get_confidence_color(confidence)};'>
                        {content['detected_condition']}: {result}
                    </h3>
                    <p>Confidence: {confidence*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
        elif disease_type in ['Skin Cancer', 'त्वचा का कैंसर']:
            prediction_data,pred,pred_score = predict_skin_cancer(image, content)
            confidence = prediction_data['confidence']
            result= prediction_data['result']
            with st.container():
                st.markdown(f"""
                <div style='background-color: rgba(255, 255, 255, 0.7); padding: 20px; border-radius: 10px;'>
                    <h3 style='color: {get_confidence_color(confidence)};'>
                        {content['detected_condition']}: {result}
                    </h3>
                    <p>Confidence: {confidence*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            probs_df = pd.DataFrame({
                'Condition': pred.values(),
                'Probability':  pred_score[0]*100
            })
            fig = px.bar(probs_df, x='Condition', y='Probability',
                        title='Probability Distribution of Conditions')
            st.plotly_chart(fig)
        
            

                        
            # Display result with a modern card design
            with st.container():
                st.markdown(f"""
                <div style='background-color: rgba(255, 255, 255, 0.7); padding: 20px; border-radius: 10px;'>
                    <h3 style='color: {get_confidence_color(confidence)};'>
                        {content['detected_condition']}: {result}
                    </h3>
                    <p>Confidence: {confidence*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
        elif disease_type in ["Lung Cancer", "फेफड़ों का कैंसर"]:
            prediction_data,pred,pred_score = lung(image)
            confidence = prediction_data['confidence']
            result= prediction_data['result']
            with st.container():
                st.markdown(f"""
                <div style='background-color: rgba(255, 255, 255, 0.7); padding: 20px; border-radius: 10px;'>
                    <h3 style='color: {get_confidence_color(confidence)};'>
                        {content['detected_condition']}: {result}
                    </h3>
                    <p>Confidence: {confidence*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            probs_df = pd.DataFrame({
                'Condition': pred.values(),
                'Probability':  pred_score[0]*100
            })
            fig = px.bar(probs_df, x='Condition', y='Probability',
                        title='Probability Distribution of Conditions')
            st.plotly_chart(fig)


            


        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

# Enhanced content dictionary
def get_content(language):
    if language == "Hindi":
        return {
            "app_info": "ऐप की जानकारी",
            "disease_prediction": "रोग की भविष्यवाणी प्रणाली",
            "select_disease": "वह रोग चुनें जिसकी आप भविष्यवाणी करवाना चाहते हैं:",
            "upload_image": "छवि अपलोड करें",
            "developers": "डेवलपर्स के बारे में",
            "disease_options": "हमारे ऐप द्वारा पूर्वानुमानित रोग",
            "switch_button": "Switch to English",
            "success_upload": "छवि सफलतापूर्वक अपलोड हो गई।",
            "loading_model": "मॉडल लोड हो रहा है",
            "tuberculosis_result": "तपेदिक की {prediction}% संभावना है",
            "tuberculosis_normal": "तपेदिक की संभावना बहुत कम है, सामान्य",
            "pneumonia_result": "न्यूमोनिया की {prediction}% संभावना है",
            "pneumonia_normal": "न्यूमोनिया की संभावना बहुत कम है, सामान्य",
            "detected_condition": "पता लगाई गई स्थिति",
            "prediction_history": "भविष्यवाणी इतिहास",
            "clear_history": "इतिहास साफ़ करें",
            "about_section": "हमारे बारे में",
            "Lung Cancer": "फेफड़ों का कैंसर",
            "Skin Cancer": "त्वचा का कैंसर",
            "Breast Cancer": "स्तन कैंसर",
            "Skin Condition": "त्वचा की स्थिति",
            "Tuberculosis": "तपेदिक",
            "Pneumonia": "न्यूमोनिया"
        }
    else:
        return {
            "app_info": "APP INFORMATION",
            "disease_prediction": "Disease Prediction System",
            "select_disease": "Select the disease you want to get predicted:",
            "upload_image": "Upload an Image",
            "developers": "ABOUT DEVELOPERS",
            "disease_options": "Diseases our app can predict",
            "switch_button": "हिन्दी में स्विच करें",
            "success_upload": "Image successfully uploaded.",
            "loading_model": "Loading Model",
            "tuberculosis_result": "There is {prediction}% chance of Tuberculosis",
            "tuberculosis_normal": "There is very less chance of Tuberculosis, Normal",
            "pneumonia_result": "There is {prediction}% chance of Pneumonia",
            "pneumonia_normal": "There is very less chance of Pneumonia, Normal",
            "detected_condition": "Detected Condition",
            "prediction_history": "Prediction History",
            "clear_history": "Clear History",
            "about_section": "About Us",
            "Lung Cancer": "Lung Cancer",
            "Skin Cancer": "Skin Cancer",
            "Breast Cancer": "Breast Cancer",
            "Skin Condition": "Skin Condition",
            "Tuberculosis": "Tuberculosis",
            "Pneumonia": "Pneumonia"
        }


# Enhanced CSS styling
page_bg_img = """
<style>
    /* Main container styling */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        background-size: cover;
        color: white;
    }
    
    /* Content container styling */
    [data-testid="stMainBlockContainer"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Header styling */
    .stAppHeader {
        background: rgba(0, 0, 0, 0.5);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 2rem;
        font-weight: bold;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Card styling */
    .card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #2196F3, #21CBF3);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(33, 150, 243, 0.4);
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed rgba(255, 255, 255, 0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Link styling */
    a {
        color: #21CBF3;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    
    a:hover {
        color: #2196F3;
        text-decoration: underline;
    }
</style>
"""
model_files = {
    'Lung Cancer': 'lung_cancer_model.joblib',
    'Skin Cancer': 'models/model2.joblib',
    'Breast Cancer': 'breast_cancer_model.joblib',
    'Skin Condition': 'models/model2.joblib',
    'Tuberculosis':'tb/models/model.joblib',
    'फेफड़ों का कैंसर': 'lung_cancer_model.joblib',
    'त्वचा का कैंसर': 'models/model2.joblib',
    'स्तन कैंसर': 'breast_cancer_model.joblib',
    'त्वचा की स्थिति': 'models/model2.joblib',
    'तपेदिक':'tb/models/model.joblib'
}
# Main application code
st.markdown(page_bg_img, unsafe_allow_html=True)
content = get_content(st.session_state.language)

# Language toggle button
if st.sidebar.button(content["switch_button"]):
    st.session_state.language = "Hindi" if st.session_state.language == "English" else "English"
    
content = get_content(st.session_state.language)

# Sidebar content
with st.sidebar:
    st.header(content["app_info"])
    st.markdown("""
    <div class="card">
        <p>Early disease detection using advanced image analysis and machine learning.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.header(content["developers"])
    developers = [
        {"name": "Vansh Khatri", "linkedin": "https://www.linkedin.com/in/vansh-khatri-3589b1282/"},
        {"name": "Hiyansh Chandel", "linkedin": "https://www.linkedin.com/in/hiyansh-chandel-476203296/"},
        {"name": "Shehzan", "linkedin": "https://www.linkedin.com/in/md-shehzan-86003128b/"},
        {"name": "Jatin Agrawal", "linkedin": "https://www.linkedin.com/in/jatinagrawal-py/"}
    ]
    
    for dev in developers:
        st.markdown(f"""
        <div class="card">
            <a href="{dev['linkedin']}" target="_blank">{dev['name']}</a>
        </div>
        """, unsafe_allow_html=True)

# Main content
st.markdown(f'<div class="stAppHeader">{content["disease_prediction"]}</div>', unsafe_allow_html=True)

# Disease selection
st.markdown(f"<h2>{content['select_disease']}</h2>", unsafe_allow_html=True)
disease_type = st.selectbox('', [
    content["Lung Cancer"],
    content["Skin Cancer"],
    content["Breast Cancer"],
    content["Skin Condition"],
    content["Tuberculosis"],
    content["Pneumonia"]
])

# Image upload
st.markdown(f"<h2>{content['upload_image']}</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader('', type=['jpg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=content['success_upload'], use_column_width=True)
    predict_disease(image, disease_type, model_files, content)

# Display prediction history
if st.session_state.history:
    st.header(content["prediction_history"])
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)
    
    if st.button(content["clear_history"]):
        st.session_state.history = []
        st.experimental_rerun()