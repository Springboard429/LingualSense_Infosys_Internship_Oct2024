from flask import Flask, render_template, request
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize the Flask app
app = Flask(__name__)

# Load the GRU model
gru_model = tf.keras.models.load_model(r"D:\projects\Infosys_LingualSense\language_detection_model_GRU.h5")

# Load the tokenizer and label encoder
with open(r"D:\projects\Infosys_LingualSense\tokenizer_GRU.pkl", 'rb') as f:
    tokenizer_gru = pickle.load(f)

with open(r"D:\projects\Infosys_LingualSense\label_encoder_GRU.pkl", 'rb') as f:
    label_encoder_gru = pickle.load(f)

# Define prediction function for GRU
def predict_language_paragraph(text):
    # Preprocess the paragraph
    seq = tokenizer_gru.texts_to_sequences([text])  # Convert text to sequence
    padded_seq = pad_sequences(seq, maxlen=100, padding='post', truncating='post')  # Pad/truncate

    # Make predictions
    prediction = gru_model.predict(padded_seq)
    predicted_label = label_encoder_gru.inverse_transform([prediction.argmax()])
    return predicted_label[0]

# Define routes
@app.route("/", methods=["GET", "POST"])
def home():
    detected_language = None
    if request.method == "POST":
        input_paragraph = request.form.get("paragraph")
        if input_paragraph.strip():
            # Predict language for the full paragraph
            detected_language = predict_language_paragraph(input_paragraph)
        else:
            detected_language = "Please enter a paragraph to predict!"
    
    return render_template("index.html", detected_language=detected_language)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
