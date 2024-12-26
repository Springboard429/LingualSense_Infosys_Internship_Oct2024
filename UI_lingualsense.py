import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
import numpy as np
import re
from langdetect import detect, DetectorFactory
import spacy

# To ensure consistent language detection
DetectorFactory.seed = 0


def clean_text(text):
    text = re.sub(r'\\', '', text)  # remove backslashes
    text = re.sub(r'[^\w\s.,?!-]', '', text)  # remove special characters
    text = re.sub(r'\s+', ' ', text)  # remove multiple spaces with single space
    text = re.sub(r'\d+', '', text)  # remove any numbers
    text = text.lower()
    return text

def load_resources():
    try:
        tokenizer = load('tokenizer.joblib')
        label_encoding = load('label_encoder.joblib')
        model = tf.keras.models.load_model('gru_model.h5')
        return tokenizer, label_encoding, model
    except FileNotFoundError as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

def sentiment_analysis(texts, tokenizer, encoder, model):
    language_list = []  # List to store detected languages and sentiments

    for text in texts:
        try:
            language_detected = detect(text)  # Detect the language of the text
        except:
            language_detected = "unknown"  # In case language detection fails
    
        # Clean the text for sentiment analysis
        process_text = clean_text(text)
        sequences = tokenizer.texts_to_sequences([process_text])
        padded_sequence = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
    
        # Predict sentiment
        prediction = model.predict(padded_sequence, verbose=0)
        predict_label_index = np.argmax(prediction)
        predict_class_label = encoder.inverse_transform([predict_label_index])[0]
        
        # Append the results as a tuple
        language_list.append((text, language_detected, predict_class_label))

    return language_list

def split_sentence(text):
    sentences = re.split(r'(?<=[.!?।؟！？।｡。⸮]) +', text)
    return sentences


# --------------------------------------Streamlit App-------------------------------------------------------------------
st.title("Sentiment Analysis Web App")

user_input = st.text_area("Enter your text here: ", height=150)

model_type = st.selectbox("Select model", ["LSTM", "GRU"])

if st.button("Analyze Language"):
    if user_input:
        with st.spinner("Analyzing..."):
            # Split the input into multiple lines
            user_texts = split_sentence(user_input)

            # Load resources
            tokenizer, encoder, gru_model = load_resources()

            # Perform sentiment analysis
            results = sentiment_analysis(user_texts, tokenizer, encoder, gru_model)

            st.write("### Prediction")

            for original_text, language, sentiment in results:
                st.write(f"**Sentence:** {original_text}")
                st.write(f"**Predicted language:** {language.capitalize()}")
                st.write(f"**Sentiment:** {sentiment}")
    else:
        st.warning("Please enter some text to analyze.")
