import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Load the GRU model
gru_model = tf.keras.models.load_model('GRU_NEW_model.h5')

# Load the tokenizer and label encoder
with open('tokenizer_gru_new.pkl', 'rb') as f:
    tokenizer_gru_new = pickle.load(f)

with open('label_encoder_gru_new.pkl', 'rb') as f:
    label_encoder_gru_new = pickle.load(f)

# Define prediction function for GRU
def predict_language_paragraph_simple(text):
    # Basic rule-based sentence splitting
    sentences = re.split(r'[.?!]', text)  # Split on ., ?, or !
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]  # Remove empty sentences
    language_predictions = []

    for sentence in sentences:
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
st.title(" LingualSense-Natural Language Detection App")

st.write(
    """
        Welcome to the **Language Detection App**! This tool leverages a GRU-based model to identify the language of each sentence 
        in a paragraph you provide. Simply input your text below to discover the languages detected for every sentence!

    """
)

# Text input for paragraph
st.subheader("Enter Your Paragraph Below:")
input_paragraph = st.text_area("Enter the Text here:", "")

if st.button("Predict Languages"):
    if input_paragraph.strip():
        # Predict languages for the full paragraph
        detected_languages = predict_language_paragraph_simple(input_paragraph)

        if detected_languages:
            st.write("### Detected Languages by Sentence:")
            for sentence, language in detected_languages:
                st.write(f"- **Sentence:** {sentence}")
                st.write(f"  - **Language:** {language}")
        else:
            st.error("No valid sentences were detected. Please try again!")
    else:
        st.error("Please enter a paragraph to predict!")
