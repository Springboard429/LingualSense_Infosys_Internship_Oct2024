import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the GRU model
gru_model = tf.keras.models.load_model(r"D:\projects\Infosys_LingualSense\language_detection_model_GRU.h5")

# Load the tokenizer and label encoder
with open(r"D:\projects\Infosys_LingualSense\tokenizer_GRU.pkl", 'rb') as f:
    tokenizer_gru = pickle.load(f)

with open(r"D:\projects\Infosys_LingualSense\label_encoder_GRU.pkl", 'rb') as f:
    label_encoder_gru = pickle.load(f)

# Define prediction function for GRU
def predict_language_paragraph(text):
    # Preprocess the paragraph
    seq = tokenizer_gru.texts_to_sequences([text])  # Convert text to sequence
    padded_seq = pad_sequences(seq, maxlen=100, padding='post', truncating='post')  # Pad/truncate

    # Make predictions
    prediction = gru_model.predict(padded_seq)
    predicted_label = label_encoder_gru.inverse_transform([prediction.argmax()])
    return predicted_label[0]

# Streamlit UI
st.title("Language Detection App (GRU Model)")
st.write("Detect the language of a paragraph using a GRU-based deep learning model!")

# Text input for paragraph
st.subheader("Input a Paragraph")
input_paragraph = st.text_area("Enter your paragraph here:", "")

if st.button("Predict Language"):
    if input_paragraph.strip():
        # Predict language for the full paragraph
        detected_language = predict_language_paragraph(input_paragraph)
        st.success(f"The detected language is: **{detected_language}**")
    else:
        st.error("Please enter a paragraph to predict!")

# About section
st.sidebar.title("About")
st.sidebar.write(
    """
This app uses a GRU-based deep learning model for language detection.
It supports detecting the language for full paragraphs.
"""
)
