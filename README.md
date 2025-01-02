# LingualSense_Infosys_Internship_Oct2024
To build a model that can automatically identify the language of a given text. Language identification is essential for various applications, including machine translation, multilingual document tracking, and electronic devices (e.g., mobiles, laptops).

# LingualSense: Deep Learning for Language Detection Across Texts

LingualSense is a deep learning project for classifying text languages. This README provides step-by-step instructions from data analysis to deployment.

---

## Steps to Follow

### 1. **Exploratory Data Analysis (EDA)**  
- Perform EDA to analyze your dataset.  
- Check the distribution of languages and clean any irregularities in the dataset.  

### 2. **Data Preprocessing**  
- Tokenize the text data and pad sequences to a uniform length for model compatibility.  
- Save the tokenizer and label encoder for future use in the app.  

### 3. **Model Building**  
- Use a GRU-based model for text classification.  
- Train the model using tokenized and padded sequences.  
- Save the trained model as `gru_model.h5`.  

### 4. **Streamlit Application Development**  
- Create a `Streamlit` app for real-time predictions.  
- Include input text areas, model loading, and prediction functionality.  
- Add a styled user interface for better interaction.  

### 5. **Setup Environment**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/Springboard429/LingualSense_Infosys_Internship_Oct2024.git
   cd LingualSense

2. **Create a virtual environment:**

   - **Windows:**
     ```bash
     python -m venv lingualsense_env
     lingualsense_env\Scripts\activate
     ```

   - **Mac/Linux:**
     ```bash
     python -m venv lingualsense_env
     source lingualsense_env/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
4. Place the following files in the project directory:
   ```bash
    gru_model.h5
    tokenizer.joblib
    label_encoder.joblib

### 6. **Run the Streamlit App**
- Execute the following command:
  ```bash
  streamlit run app.py
Open the local URL (e.g., http://localhost:XXXX) to access the app.

### 7. **Usage**
- Input text in the text area.
- Click "Detect Languages" to get the predicted language of the text.


