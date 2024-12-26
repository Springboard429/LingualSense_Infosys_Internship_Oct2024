# predict.py
import joblib
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Function to load the tokenizer, label encoder, and model
def load_resources():
    tokenizer = joblib.load('./model/tokenizer.joblib')
    encoder = joblib.load('./model/label_encoder.joblib')
    model = load_model('./model/gru.h5')
    return tokenizer, encoder, model


# Function to clean the input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text


# Function to preprocess the sentence
def preprocess_sentence(sentence, tokenizer, max_length):
    cleaned_sentence = clean_text(sentence)
    sequence = tokenizer.texts_to_sequences([cleaned_sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    return padded_sequence


# Function to predict the class of the sentence
def predict_class(sentence, tokenizer, encoder, model, max_length=100):
    input_data = preprocess_sentence(sentence, tokenizer, max_length)
    prediction = model.predict(input_data)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = encoder.inverse_transform([predicted_class_index])[0]
    return predicted_class_index, predicted_class_label