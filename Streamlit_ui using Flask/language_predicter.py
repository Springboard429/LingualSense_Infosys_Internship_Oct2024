from flask import Flask, render_template, request
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize

app = Flask(__name__)

# Load the GRU model, tokenizer, and label encoder (update path to your model and files)
desktop_path = r"C:\Users\TAUFIQUE\Desktop"
model_filename = desktop_path + r"\gru-model.h5"
tokenizer_filename = desktop_path + r"\tokenizer.pkl"
label_encoder_filename = desktop_path + r"\label_encoder.pkl"

# Load the model and tokenizer files
gru_model = tf.keras.models.load_model(model_filename)
with open(tokenizer_filename, 'rb') as f:
    tokenizer_gru_new = pickle.load(f)
with open(label_encoder_filename, 'rb') as f:
    label_encoder_gru_new = pickle.load(f)

# Prediction function that returns both the sentence and its detected language
def predict_language_paragraph(text):
    sentences = sent_tokenize(text)
    language_predictions = []

    for sentence in sentences:
        if sentence.strip():
            seq = tokenizer_gru_new.texts_to_sequences([sentence])
            padded_seq = pad_sequences(seq, maxlen=100, padding='post', truncating='post')

            prediction = gru_model.predict(padded_seq)
            predicted_label = label_encoder_gru_new.inverse_transform([prediction.argmax()])
            language_predictions.append((sentence, predicted_label[0]))  # Return sentence and its predicted language

    return language_predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        input_paragraph = request.form.get('paragraph')
        if input_paragraph:
            # Get predicted languages for each sentence
            detected_languages = predict_language_paragraph(input_paragraph)
            prediction = detected_languages  # Store list of (sentence, language)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

