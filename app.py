# import streamlit as st
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.text import tokenizer_from_json
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import numpy as np
# import pickle
# import json
# from langdetect import detect

# # Load the trained model
# model = load_model("gru_language_detection_model.h5")

# # Define the max sequence length (match it with your training setup)
# MAX_SEQUENCE_LENGTH = 100

# # Load the tokenizer
# with open("tokenizer.json", "r") as f:
#     tokenizer_json = json.load(f)
#     tokenizer = tokenizer_from_json(tokenizer_json)

# # Load the label encoder
# with open("label_encoder.pkl", "rb") as f:
#     label_encoder = pickle.load(f)

# # Preprocessing function
# def preprocess_input(input_text):
#     tokenized_input = tokenizer.texts_to_sequences([input_text])
#     padded_input = pad_sequences(tokenized_input, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
#     return padded_input

# # Prediction function
# def predict_language(input_text):
#     preprocessed_input = preprocess_input(input_text)
#     predictions = model.predict(preprocessed_input)
#     predicted_language = label_encoder.inverse_transform([np.argmax(predictions)])
#     return predicted_language[0]

# # Multilingual prediction function
# def predict_multilingual(input_text):
#     sentences = input_text.split('.')
#     detected_languages = {}

#     for sentence in sentences:
#         if sentence.strip():
#             lang = detect(sentence)
#             predicted_lang = predict_language(sentence)

#             if predicted_lang not in detected_languages:
#                 detected_languages[predicted_lang] = []
#             detected_languages[predicted_lang].append(sentence.strip())

#     return detected_languages

# # Streamlit UI
# st.title("Multilingual Language Detection")
# st.write("Upload text to detect languages used within it.")

# input_text = st.text_area("Enter your text:", "She loves to travel. Elle rêve de découvrir de nouveaux endroits. और वह जीवन से खुश है।")

# if st.button("Detect Languages"):
#     if input_text.strip():
#         results = predict_multilingual(input_text)
#         st.write("Detected Languages and Sentences:")
#         for language, sentences in results.items():
#             st.write(f"**{language}**:")
#             for sentence in sentences:
#                 st.write(f"- {sentence}")
#     else:
#         st.warning("Please enter some text.")
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import json
from langdetect import detect

# Load the trained model
model = load_model("gru_language_detection_model.h5")

# Define the max sequence length
MAX_SEQUENCE_LENGTH = 100

# Load the tokenizer
with open("tokenizer.json", "r") as f:
    tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)

# Load the label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

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
    page_icon="🌍",
    layout="wide"
)

# Header Section with Styling
st.markdown("""
    <style>
        .main-header {
            font-size: 40px;
            color: #2E8B57;
            text-align: center;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .sub-header {
            font-size: 18px;
            text-align: center;
            color: #444444;
            margin-bottom: 20px;
        }
        .footer {
            font-size: 14px;
            text-align: center;
            margin-top: 50px;
            color: #888888;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌍 LingualSense 🌍</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Detect languages in your text effortlessly!</div>', unsafe_allow_html=True)

# Input Section
st.sidebar.title("Upload Text")
st.sidebar.write("Paste or type text to detect languages.")

input_text = st.sidebar.text_area("")

# Button to Start Detection
if st.sidebar.button("Detect Languages"):
    if input_text.strip():
        with st.spinner("Analyzing your text..."):
            results = predict_multilingual(input_text)
        st.success("Languages Detected!")
        
        # Results Section
        st.subheader("Detected Languages and Sentences:")
        for language, sentences in results.items():
            st.markdown(f"### {language}")
            for sentence in sentences:
                st.markdown(f"- {sentence}")
    else:
        st.sidebar.warning("Please enter some text.")

# Footer Section
st.markdown('<div class="footer">© 2024 Multilingual Language Detection App | Powered by Deep Learning</div>', unsafe_allow_html=True)

