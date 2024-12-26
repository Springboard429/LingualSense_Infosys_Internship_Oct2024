import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model_GRU = load_model('model_gru.keras')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Function to clean text
def clean_text_regex(text):
    if isinstance(text, str):
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
    return text

# Function to predict the language
def predict_language(text):
    cleaned_text = clean_text_regex(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    prediction = model_GRU.predict(padded_sequence)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_language = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_language

st.set_page_config(
    page_title="Language Prediction",  # Browser tab title
    page_icon="None",                   # Icon in the browser tab
    layout="wide",                    # Set the app layout to wide
    initial_sidebar_state="auto"  # Sidebar starts expanded
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff;
    }
    .main-title {
            width: 100%;
            height: 100px; /* Adjust height as needed */
            top: 0px
            margin-top: 0px;
            background-color: #f5f5dc; /* Light yellow background */
            display: flex;
            justify-content: center; /* Center the text horizontally */
            align-items: center; /* Center the text vertically */
            font-size: 48px; /* Adjust font size */
            margin-bottom: 40px; /* Add space below the header */
            color: #000; /* Text color */
        }
    .sidebar {
        background-color: #f5f5dc;
        padding: 15px;
        font-size: 18px;
        color: black;
        height: 100%;
        overflow-y: auto;
    }
    .main-content {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        width: 100%;
        margin: 20px auto;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
            background-color: #e3f2fd; /* Green background */
            color: black;             /* White text */
            border: none;             /* Remove border */
            border-radius: 5px;       /* Rounded corners */
            padding: 10px 20px;       /* Padding for the button */
            font-size: 16px;          /* Increase font size */
            cursor: pointer;          /* Pointer cursor on hover */
        }
        
    .stButton > button:hover {
        background-color: #A0B0B9; /* Darker green on hover */
        }
    textarea {
        width: 100%;
        height: 50%;
        margin-bottom: 20px;
        padding: 10px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Header
st.markdown('<div class="main-title">Language Prediction</div>', unsafe_allow_html=True)

# Sidebar
languages = [
    'English', 'Malayalam', 'Hindi', 'Tamil', 'Portuguese', 'French', 'Dutch',
    'Spanish', 'Greek', 'Russian', 'Danish', 'Italian', 'Turkish', 'Swedish',
    'Arabic', 'German', 'Kannada', 'Estonian', 'Thai', 'Japanese', 'Latin',
    'Urdu', 'Indonesian', 'Chinese', 'Korean', 'Pushto', 'Persian', 'Romanian'
]

st.sidebar.markdown('<div class="sidebar"><b>Languages that can predict:</b><ul>' +
                    ''.join(f'<li>{language}</li>' for language in languages) +
                    '</ul></div>', unsafe_allow_html=True)


st.header('Enter text here')
# Input Text Area
input_text = st.text_area("", height=200)

if st.button("Predict Language"):
    if input_text:
        # Predict the language
        predicted_language = predict_language(input_text)
        st.success(f"The predicted language is: {predicted_language}")
    else:
        st.warning("Please enter some text to predict the language.")

st.markdown('</div>', unsafe_allow_html=True)
