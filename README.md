### **LingualSense Natural Language Detection App**

---

#### **Project Overview**

**LingualSense** is a language detection application designed to identify the language of each sentence within a given text. The app leverages a GRU-based deep learning model to deliver accurate predictions for various languages. This project was developed as part of the Infosys Internship in October 2024.

---

#### **Features**

- **Multi-Language Detection**: Detects the language of each sentence in a multi-lingual paragraph.
- **Deep Learning Model**: Uses a GRU-based neural network for language prediction.
- **User-Friendly Interface**: Built with Streamlit for an intuitive, web-based user experience.
- **Efficient Sentence Tokenization**: Handles multiple sentences within a paragraph using custom tokenization logic.

---

#### **Tech Stack**

- **Framework**: Streamlit
- **Machine Learning**: TensorFlow and Keras
- **Language Processing**: Tokenizer (Keras) and custom sentence splitting with `re`.
- **Programming Language**: Python
- **Data**: Cleaned and preprocessed language dataset with text-label pairs.

---

#### **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/Springboard429/LingualSense_Infosys_Internship_Oct2024.git
   cd LingualSense_Infosys_Internship_Oct2024
   ```

2. Switch to the relevant branch:
   ```bash
   git checkout Akshara
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the GRU model and associated files are in place:
   - `GRU_NEW_model.h5`
   - `tokenizer_gru_new.pkl`
   - `label_encoder_gru_new.pkl`

5. Run the application:
   ```bash
   streamlit run app.py
   ```

---

#### **Usage**

1. Open the Streamlit app in your browser.
2. Paste a paragraph containing one or multiple sentences in different languages.
3. Click on **Predict Languages** to see the detected language for each sentence.

---

#### **Directory Structure**

```plaintext
LingualSense_Infosys_Internship_Oct2024/
├── app.py                      # Streamlit application script
├── GRU_NEW_model.h5            # Pre-trained GRU model for language detection
├── tokenizer_gru_new.pkl       # Tokenizer for text preprocessing
├── label_encoder_gru_new.pkl   # Label encoder for language labels
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
```

---

#### **Examples**

- **Input**: 
  ```plaintext
  The quick brown fox jumps over the lazy dog.
  El rápido zorro marrón salta sobre el perro perezoso.
  ```

- **Output**:
  ```plaintext
  Sentence: The quick brown fox jumps over the lazy dog.
  Language: English

  Sentence: El rápido zorro marrón salta sobre el perro perezoso.
  Language: Spanish
  ```

---

#### **Known Issues**

- Accuracy may degrade for very complex paragraphs or rarely encountered languages.
- Multi-language paragraphs might result in occasional misclassifications due to dataset limitations.

---

#### **Contributors**

- Akshara

---

#### **License**

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

#### **Future Work**

- Expand the dataset to include more languages and diverse examples.
- Improve sentence splitting logic for better handling of complex inputs.
- Introduce support for real-time language detection in live chats or documents.

---

Feel free to contribute to this project by submitting pull requests or raising issues in the repository.
