import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import spacy
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
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Function to split text into sentences
def custom_sentence_splitter(text):
    # Define a pattern to match common sentence-ending punctuation
    sentence_endings = r'[.!?。！؟]'
    sentences = re.split(sentence_endings, text)
    
    # Remove any empty strings from the result
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    return sentences

# Function to predict the language of a single sentence
def predict_language(text):
    cleaned_text = clean_text_regex(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    prediction = model_GRU.predict(padded_sequence)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_language = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_language

# Function to predict languages in a structured paragraph
def predict_languages_in_paragraph(paragraph):
    # Split the paragraph into sentences using the enhanced function
    sentences = custom_sentence_splitter(paragraph)
    
    # List to store results
    language_results = []
    
    for i, sentence in enumerate(sentences):
        if sentence.strip():  # Ignore empty sentences
            predicted_language = predict_language(sentence)
            language_results.append((i + 1, sentence, predicted_language))
    
    return language_results

# Streamlit UI
st.set_page_config(page_title="Multi-Language Detection", layout="wide")


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

        .paragraph-label {
        font-size: 20px; /* Increase font size */
        color: #2c3e50; /* Change text color (dark blue) */
        font-weight: bold; /* Make the text bold */
        margin-bottom: 10px; /* Add spacing below the text */
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

st.markdown('<div class="main-title">Multi Language Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="paragraph-label">Enter a paragraph with sentences in different languages,and the app will identify the language of each sentence.</div>', unsafe_allow_html=True)

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


st.subheader("Enter the text here:")

# Text area for input
input_paragraph = st.text_area("", height=200)

if st.button("Detect Languages"):
    if input_paragraph:
        # Detect languages in the paragraph
        detected_languages = predict_languages_in_paragraph(input_paragraph)
        
        # Display the results
        detected_sentences = '<div class="detected-sentences">'
        for sentence_number, sentence, language in detected_languages:
            detected_sentences += f'<p>Sentence {sentence_number}: "{sentence}" - Detected Language: <b>{language}</b></p>'
        detected_sentences += '</div>'
        
        # Inject custom CSS for styling
        st.markdown(
            """
            <style>
            .detected-sentences {
                font-size: 16px; /* Adjust font size as needed */
                color: white; /* Change text color to white */
                font-weight: normal; /* Keep it readable */
                line-height: 1.5; /* Add spacing between lines for clarity */
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Display the styled detected sentences
        st.markdown(detected_sentences, unsafe_allow_html=True)
    else:
        st.warning("Please enter a paragraph to detect languages.")