import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# Load the GRU model
gru_model = tf.keras.models.load_model(r"C:\Users\prama\OneDrive\Desktop\LingualSense\GRU_NEW_model.h5")

# Load the tokenizer and label encoder
with open(r"C:\Users\prama\OneDrive\Desktop\LingualSense\tokenizer_gru_new.pkl", 'rb') as f:
    tokenizer_gru_new = pickle.load(f)

with open(r"C:\Users\prama\OneDrive\Desktop\LingualSense\label_encoder_gru_new.pkl", 'rb') as f:
    label_encoder_gru_new = pickle.load(f)

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
st.title("🌍 LingualSense-Language Detection App")

st.write(
    """
    Welcome to the **Language Detection App**! This app uses a GRU-based model to detect the language of each sentence 
    in a given paragraph. Enter a paragraph below and see the detected languages for each sentence!
    """
)

# Text input for paragraph
st.subheader("Enter Your Paragraph Below:")
input_paragraph = st.text_area("Type or paste your paragraph here:", "")

if st.button("Predict Languages"):
    if input_paragraph.strip():
        # Predict languages for the full paragraph
        detected_languages = predict_language_paragraph(input_paragraph)

        if detected_languages:
            st.write("### Detected Languages by Sentence:")
            for sentence, language in detected_languages:
                st.write(f"- **Sentence:** {sentence}")
                st.write(f"  - **Language:** {language}")
        else:
            st.error("No valid sentences were detected. Please try again!")
    else:
        st.error("Please enter a paragraph to predict!")

# Sidebar: About
st.sidebar.title("About")
st.sidebar.write(
    """
    **Language Detection App**  
    - Built with Streamlit and TensorFlow  
    - Uses a GRU-based deep learning model to detect languages.  
    - Detects the language for each sentence in a paragraph.  
    """
)

st.sidebar.write("Developed by Agniv Pramanick")