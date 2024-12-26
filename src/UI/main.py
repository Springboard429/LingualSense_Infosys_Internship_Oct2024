import streamlit as st
from utils import load_resources, predict_class
from tensorflow import keras


# Function to load resources only once using Streamlit's session state
def load_model_resources():
    if 'tokenizer' not in st.session_state:
        # Load the resources (tokenizer, encoder, model) into session state
        st.session_state.tokenizer, st.session_state.encoder, st.session_state.model = load_resources()


# Load model resources on app start
load_model_resources()

# Set page configuration
st.set_page_config(
    page_title="LingualSense",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Sidebar menu
st.sidebar.markdown(
    """
    <div style="padding: 20px; background-color: #292b2c; border-radius: 10px; color: #f0f0f0;">
        <h3 style="text-align: center;">About LingualSense</h3>
        <p style="font-size: 14px; text-align: justify; color: #dcdcdc;">
        LingualSense is a tool that detects the language of a given text by analyzing its linguistic 
        features and comparing them to known language models. Developed using advanced machine learning 
        models, LingualSense ensures accurate and efficient language classification. 
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Add background styling
# Add professional styling
st.markdown(
    """
    <style>
        /* General body styling */
        body {
            background-color: #1e1e1e; /* Dark gray background */
            font-family: "Times New Roman", serif;
        }
        .main {
            background-color: #292b2c; /* Slightly lighter dark gray */
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            color: #f0f0f0; /* Light text color */
        }
        h1 {
            color: #ffffff; /* White text */
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 20px;
        }
        p {
            color: #dcdcdc; /* Lighter gray text */
            text-align: center;
            font-size: 1.2em;
        }
        .stTextArea>div>textarea {
            background-color: #3c3f41; /* Dark input box */
            color: #ffffff; /* White text */
            border: 1px solid #555555; /* Subtle border */
            border-radius: 5px;
        }
        .stButton>button {
            background-color: #4caf50; /* Green button */
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049; /* Darker green on hover */
        }
        .result-card {
            background-color: #333333;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .result-card h3, .result-card h4 {
            color: #f0f0f0; /* Light text */
            text-align: center;
        }
        
    </style>
    """,
    unsafe_allow_html=True
)

# Add title and sub-header
st.markdown("<h1>🔍 LingualSense</h1>", unsafe_allow_html=True)
st.markdown("<p>LingualSense is a tool that detects the language of a given text by analyzing its linguistic "
            "features and comparing them to known language models.</p>", unsafe_allow_html=True)

# Input text area
input_sentences = st.text_area(
    "Input Text",
    height=100,
    placeholder="Type your text",
)

# Predict button
if st.button("Classify Sentence", key="predict_button", help="Click to predict the class of the input sentences"):
    if input_sentences:
        # Load resources from session state
        tokenizer = st.session_state.tokenizer
        encoder = st.session_state.encoder
        model = st.session_state.model

        # Split sentences by dot and strip any extra whitespace
        sentences = [sentence.strip() for sentence in input_sentences.split('.') if sentence.strip()]

        # Predict and display results for each sentence
        for sentence in sentences:
            predicted_class_index, predicted_class_label = predict_class(sentence, tokenizer, encoder, model)
            st.markdown(
                f"""
                <div style="background-color: #444444; padding: 20px; border-radius: 10px; margin-top: 20px; box-shadow:
                 2px 2px 10px rgba(0, 0, 0, 0.5);">
                    <h3 style="text-align: center; color: white;">Sentence: "{sentence}"</h3>
                    <h4 style="text-align: center; color: white;">Language: <strong>{predicted_class_label}</strong>
                    </h4>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("Please enter sentences to predict their classes.", icon="⚠️")

# Add some spacing
st.markdown("<br><br>", unsafe_allow_html=True)
