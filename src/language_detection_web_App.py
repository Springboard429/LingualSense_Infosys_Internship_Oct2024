import re
import pickle
import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load pre-trained model and tokenizer
model_path = 'C:/Users/thota/Desktop/LingualSenseOct2024/LingualSense_Infosys_Internship_Oct2024/src/trainedmodel.sav'
tokenizer_path = 'C:/Users/thota/Desktop/LingualSenseOct2024/LingualSense_Infosys_Internship_Oct2024/src/savedtokenizer.pkl'

try:
    loaded_model = pickle.load(open(model_path, 'rb'))
    tokenizer = pickle.load(open(tokenizer_path, 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading model or tokenizer: {e}")
    st.stop()

# Cleaning text
def clean_text(text):
    text = re.sub(r'\\', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Label Encoding
languagesforencode = [
    'English', 'French', 'Spanish', 'Russian', 'Dutch', 'Arabic', 'Turkish',
    'Tamil', 'Hindi', 'Romanian', 'Persian', 'Pushto', 'Swedish', 'Estonian',
    'Korean', 'Chinese', 'Portuguese', 'Indonesian', 'Urdu', 'Latin', 'Japanese',
    'Thai', 'Portuguese', 'Italian', 'Swedish', 'Malayalam', 'German', 'Danish',
    'Kannada', 'Greek'
]

label_encoder = LabelEncoder()
label_encoder.fit(languagesforencode)

# Function for Prediction
def predict_languages(texts, tokenizer, max_length, label_encoder):
    predictions = []
    for text in texts:
        cleaned_text = clean_text(text)
        sequences = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequences = pad_sequences(sequences, maxlen=max_length)
        probabilities = loaded_model.predict(padded_sequences)[0]
        predicted_language = label_encoder.inverse_transform([probabilities.argmax()])[0]
        predictions.append((predicted_language, probabilities.max()))
    return predictions

# Streamlit App
def main():
    st.title('LingualSense Web App')

    # Dynamic input fields
    num_inputs = st.slider('Number of texts to input', 1, 5, 1)
    texts = [st.text_input(f'Input Text {i+1}') for i in range(num_inputs)]

    if st.button('Predict Language'):
        if all(texts):  # Ensure no empty input
            max_length = 120  # Replace with actual max length used in training
            results = predict_languages(texts, tokenizer, max_length, label_encoder)
            for i, (language, confidence) in enumerate(results):
                st.success(f"Text {i+1}: Predicted Language - {language} (Confidence: {confidence:.2f})")
        else:
            st.warning("Please fill all input fields.")

if __name__ == '__main__':
    main()
