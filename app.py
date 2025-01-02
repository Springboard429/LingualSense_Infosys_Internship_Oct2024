import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib
from langdetect import detect

# Load the trained model
model = load_model("gru_model.h5")

# Define the max sequence length
MAX_SEQUENCE_LENGTH = 100

# Load the tokenizer
tokenizer = joblib.load("tokenizer.joblib")

# Load the label encoder
label_encoder = joblib.load("label_encoder.joblib")

# Preprocessing function
def preprocess_input(input_text):
    tokenized_input = tokenizer.texts_to_sequences([input_text])
    padded_input = pad_sequences(tokenized_input, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    return padded_input

# Prediction function
def predict_language(input_text):
    preprocessed_input = preprocess_input(input_text)
    predictions = model.predict(preprocessed_input)
    predicted_language = label_encoder.inverse_transform([np.argmax(predictions)])
    return predicted_language[0]

# Multilingual prediction function
def predict_multilingual(input_text):
    sentences = input_text.split('.')
    detected_languages = {}

    for sentence in sentences:
        if sentence.strip():
            lang = detect(sentence)
            predicted_lang = predict_language(sentence)

            if predicted_lang not in detected_languages:
                detected_languages[predicted_lang] = []
            detected_languages[predicted_lang].append(sentence.strip())

    return detected_languages

# Streamlit UI
st.set_page_config(
    page_title="LingualSense",
    page_icon="🌐",
    layout="wide"
)

# Custom CSS for Styling
st.markdown("""
    <style>
        .main-header {
            font-size: 42px;
            color: #1E90FF;
            text-align: center;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .sub-header {
            font-size: 20px;
            text-align: center;
            color: #555555;
            margin-bottom: 25px;
        }
        .sidebar-title {
            font-size: 18px;
            font-weight: bold;
            color: #2E8B57;
            margin-bottom: 10px;
        }
        .footer {
            font-size: 14px;
            text-align: center;
            margin-top: 50px;
            color: #888888;
        }
        .result-header {
            font-size: 22px;
            color: #4CAF50;
            font-weight: bold;
            margin-top: 20px;
        }
        .sentence {
            font-size: 16px;
            color: #333333;
            margin-left: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌐 LingualSense</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Effortlessly detect languages in your text with AI!</div>', unsafe_allow_html=True)

# Sidebar Input Section
st.sidebar.markdown('<div class="sidebar-title">Upload or Type Text</div>', unsafe_allow_html=True)
input_text = st.sidebar.text_area("Paste your text below:")

# Detect Languages Button
if st.sidebar.button("Detect Languages"):
    if input_text.strip():
        with st.spinner("Analyzing your text..."):
            results = predict_multilingual(input_text)
        st.success("Languages Detected Successfully!")
        
        # Display Results
        st.markdown('<div class="result-header">Detected Languages and Corresponding Sentences:</div>', unsafe_allow_html=True)
        for language, sentences in results.items():
            st.markdown(f"### {language}")
            for sentence in sentences:
                st.markdown(f'<div class="sentence">- {sentence}</div>', unsafe_allow_html=True)
    else:
        st.sidebar.warning("Please enter some text.")

# Footer Section
st.markdown('<div class="footer">© 2024 LingualSense | Powered by AI and Deep Learning</div>', unsafe_allow_html=True)
