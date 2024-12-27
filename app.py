import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import json
import joblib

# Set Page Configuration for Better UI
st.set_page_config(page_title="LingoDetect - Your Multilingual Companion", page_icon="🌍", layout="wide")

# Custom CSS for Styling
st.markdown(
    """
    <style>
        .main {
            background-color: #6A1B9A;
            font-family: 'Arial', sans-serif;
        }
        .title-text {
            text-align: center;
            font-size: 2.8rem;
            color: #FFFFFF;
            font-weight: bold;
        }
        .subtitle-text {
            text-align: center;
            font-size: 1.2rem;
            color: white;
        }
        .stTextArea > label > div > span {
            font-size: 1rem;
            color: white;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 1rem;
            padding: 8px 16px;
            border-radius: 8px;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .language-result {
            color: white;
            font-size: 1rem;
            text-align: center;
        }
        hr {
            border: 1px solid white;
        }
        p {
            color: white;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Subtitle
st.markdown("<div class='title-text'>LingoDetect - Your Multilingual Companion 🌍</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Effortlessly Detect Languages and Break Communication Barriers!</div>", unsafe_allow_html=True)

# Load Model and Tokenizer
model = joblib.load('language_model.joblib')
with open('label_mapping.json', 'r') as f:
    label_mapping = json.load(f)

tokenizer = joblib.load('tokenizer.joblib')

# Input Section
st.markdown("<h3 style='color: white;'>Enter Your Text Below</h3>", unsafe_allow_html=True)
user_input = st.text_area("Paste or Type Your Text Here:", placeholder="Type in any language you like...")

# Language Prediction
if st.button("Detect Language"):
    if user_input.strip() == "":
        st.warning("Please enter some text to detect the language.")
    else:
        try:
            # Preprocess input text
            sequences = tokenizer.texts_to_sequences([user_input])
            padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen if required
            
            # Make prediction
            prediction = model.predict(padded_sequences)
            predicted_label = np.argmax(prediction, axis=1)[0]

            # Map label to language
            language = label_mapping.get(str(predicted_label), "Unknown")

            # Display Prediction
            st.markdown(f"<div class='language-result'>Detected Language: <strong>{language.upper()}</strong></div>", unsafe_allow_html=True)
        except Exception as e:
            st.error("Sorry, we couldn't detect the language. Please try again.")

# Footer
st.markdown("""
    <hr>
    <p>
        Breaking Boundaries, Connecting Cultures 🌐 | Your Gateway to Multilingual Connections 🌎
    </p>
""", unsafe_allow_html=True)
