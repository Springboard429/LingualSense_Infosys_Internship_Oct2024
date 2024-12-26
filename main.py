import streamlit as st
from utils import load_resources, predict_class

# Function to load resources only once using Streamlit's session state
def load_model_resources():
    if 'tokenizer' not in st.session_state:
        # Load the resources (tokenizer, encoder, model) into session state
        st.session_state.tokenizer, st.session_state.encoder, st.session_state.model = load_resources()

# Load model resources on app start
load_model_resources()

# Set page configuration
st.set_page_config(
    page_title="Text Classification with GRU Model",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add background styling
st.markdown(
    """
    <style>
    /* Background gradient */
    body {
        background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
        color: #000000;
    }

    /* Title styling */
    h1 {
        font-size: 3rem;
        text-align: center;
        color: #4b0082; /* Indigo color for the title */
        margin-bottom: 10px;
    }

    /* Subheader styling */
    p {
        text-align: center;
        color: #333333; /* Dark gray for the subheader */
        font-size: 1.2rem;
    }

    /* Input box styling */
    .stTextArea {
        border-radius: 8px;
        background-color: #ffffff;
        padding: 10px;
        font-size: 1.2rem;
        color: #333333;
    }

    /* Button styling */
    button[kind="primary"] {
        background-color: #ff4500 !important; /* Orange-red for the button */
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        padding: 10px 20px !important;
    }

    /* Footer styling */
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: #4b0082; /* Indigo for the footer text */
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Add title and subheader
st.markdown("<h1>Text Classification with GRU Model</h1>", unsafe_allow_html=True)
st.markdown("<p>Enter a sentence below and click 'Classify Sentence' to predict its class.</p>", unsafe_allow_html=True)

# Input text area
input_sentence = st.text_area(
    "Input Text",
    height=100,
    placeholder="Type your text here (e.g., 'If this solves the problem...')",
)

# Predict button
if st.button("Classify Sentence", key="predict_button", help="Click to predict the class of the input sentence"):
    if input_sentence:
        # Load resources from session state
        tokenizer = st.session_state.tokenizer
        encoder = st.session_state.encoder
        model = st.session_state.model

        # Get prediction
        predicted_class_index, predicted_class_label = predict_class(input_sentence, tokenizer, encoder, model)

        # Display result in a card
        st.markdown(
            f"""
            <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; margin-top: 20px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);">
                <h3 style="text-align: center; color: #333333;">The given text is <strong>{predicted_class_label}</strong></h3> <!-- Dark gray -->
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("Please enter a sentence to predict the class.", icon="⚠️")

# Add some spacing
st.markdown("<br><br>", unsafe_allow_html=True)

# Footer with branding
st.markdown(
    """
    <div class="footer">
        <hr>
        <p>Powered by LingualSense and GRU Model | <a href="https://github.com" target="_blank" style="color: #ff4500; text-decoration: underline;">GitHub</a></p> <!-- Orange-red -->
    </div>
    """,
    unsafe_allow_html=True,
)