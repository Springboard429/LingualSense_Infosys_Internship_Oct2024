import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize
import nltk
import os

# Download necessary NLTK data
nltk.download('punkt')

# Define file paths
desktop_path = r"C:\Users\TAUFIQUE\Desktop"
model_filename = os.path.join(desktop_path, "gru-model.h5")
tokenizer_filename = os.path.join(desktop_path, "tokenizer.pkl")
label_encoder_filename = os.path.join(desktop_path, "label_encoder.pkl")

# Check if files exist before loading
if os.path.exists(model_filename) and os.path.exists(tokenizer_filename) and os.path.exists(label_encoder_filename):
    # Load the GRU model
    gru_model = tf.keras.models.load_model(model_filename)

    # Load the tokenizer and label encoder
    with open(tokenizer_filename, 'rb') as f:
        tokenizer_gru_new = pickle.load(f)

    with open(label_encoder_filename, 'rb') as f:
        label_encoder_gru_new = pickle.load(f)
else:
    st.error("Model or necessary files are missing. Please ensure all files are present on the Desktop.")
    st.stop()

# Define prediction function for GRU
def predict_language_paragraph(text):
    # Tokenize the paragraph into sentences
    sentences = sent_tokenize(text)
    language_predictions = []

    for sentence in sentences:
        if sentence.strip():  # Skip empty sentences
            # Preprocess the sentence
            seq = tokenizer_gru_new.texts_to_sequences([sentence])  # Convert text to sequence
            padded_seq = pad_sequences(seq, maxlen=100, padding='post', truncating='post')  # Pad/truncate

            # Make predictions for each sentence
            prediction = gru_model.predict(padded_seq)
            predicted_label = label_encoder_gru_new.inverse_transform([prediction.argmax()])
            language_predictions.append((sentence, predicted_label[0]))

    return language_predictions

# Streamlit UI
st.set_page_config(page_title="Language Detection App", layout="wide")

# Custom style for modern look
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f1f1f1;
            color: #333;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            margin-top: 20px;
        }
        .description {
            font-size: 18px;
            color: #666;
            text-align: center;
            margin: 20px 0;
        }
        .input-textarea {
            font-size: 16px;
            font-weight: bold;
            padding: 10px;
            border-radius: 10px;
            border: 2px solid #4CAF50;
            background-color: #fff;
            width: 100%;
            max-width: 800px;
            margin: 0 auto;
        }
        .button {
            font-size: 18px;
            padding: 12px 25px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            max-width: 800px;
            margin: 20px auto;
            display: block;
        }
        .button:hover {
            background-color: #45a049;
        }
        .result {
            font-size: 18px;
            margin-top: 20px;
            color: red;
            text-align: center;
        }
        .sentence {
            font-weight: bold;
            color: white;
        }
        .language {
            font-weight: bold;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">LingualSense - Language Detection App</div>', unsafe_allow_html=True)

st.markdown('<div class="description">This app uses a GRU-based model to detect the language of each sentence in a given paragraph.</div>', unsafe_allow_html=True)

# Text input for paragraph
st.subheader("Enter Your Paragraph Below:")
input_paragraph = st.text_area("", "", height=150)

# Button to predict language
if st.button("Predict Languages"):
    if input_paragraph.strip():
        # Predict languages for the full paragraph
        detected_languages = predict_language_paragraph(input_paragraph)

        if detected_languages:
            st.write("### Detected Languages by Sentence:")
            for sentence, language in detected_languages:
                st.markdown(
                    f'<div class="result">'
                    f'<span class="sentence">Sentence:</span> {sentence} <br>'
                    f'<span class="language">Language:</span> {language}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.error("No valid sentences were detected. Please try again!")
    else:
        st.error("Please enter a paragraph to predict!")
