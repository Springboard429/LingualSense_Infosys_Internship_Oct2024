import streamlit as st
from tensorflow.keras.models import load_model
import pickle


# Load the model and preprocessors
model = load_model('language_detection_gru.h5')

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# App layout
st.title("Language Detection App")
user_input = st.text_area("Enter your text below:")

if st.button("Detect Language"):
    if user_input:
        # Transform input using TF-IDF
        input_tfidf = tfidf.transform([user_input]).toarray()

        # Predict
        prediction = model.predict(input_tfidf)
        predicted_label = label_encoder.inverse_transform([prediction.argmax()])[0]

        st.write(f"The detected language is: **{predicted_label}**")
    else:
        st.write("Please enter some text.")
