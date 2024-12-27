# LingualSense: Automatic Language Identification

## Infosys Internship Project - October 2024

### Overview
LingualSense is a machine learning model designed to automatically identify the language of a given text. Language identification plays a crucial role in applications such as:
- Machine translation
- Multilingual document tracking
- Language-based content personalization for electronic devices (e.g., mobiles, laptops)

This project explores and implements state-of-the-art techniques to achieve accurate and efficient language detection.

---

## Features
- **Multi-language support**: Detects a wide range of languages with high accuracy.
- **Scalable architecture**: Supports large datasets for training and evaluation.
- **Integration-ready**: Provides APIs and utilities for easy integration with other systems.

---

## Table of Contents
- [Getting Started](#getting-started)
- [Installation](#installation)
- [File Structure](#file-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)

---

## Getting Started
To get started with LingualSense, clone the repository and follow the instructions below to set up the environment and run the model.

---

## Installation

### Prerequisites
- Python 3.8+
- pip

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Springboard429/LingualSense_Infosys_Internship_Oct2024.git
   cd LingualSense_Infosys_Internship_Oct2024
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---
## File Structure

The repository is organized as follows:

```
LingualSense/
├── dataset/                     # Dataset files
│   ├── LingualSense_dataset[1].csv            # Combined dataset
├── models/                   # Pre-trained models
├── src/                  # Scripts for tasks
│   ├── LingualSense_DL.ipynb # Deep learning notebook
│   ├── LingualSense_ML.ipynb # Machine learning notebook
│   ├── UI/                   # User interface
│       ├── model             # Models for UI
│       ├── main.py           # Streamlit app
│       ├── utils.py          # Utility functions
├── LingualSense- Deep Learning for Language Detection Across Texts.pdf
                              # Project document
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── LICENSE                   # License file
```

---

## Dataset
The dataset used for this project contains two columns:
- **Language**: The language of the text sample (e.g., English, Spanish, French).
- **Sample Text**: A text snippet in the corresponding language.

### Example:
| Sample Text           |  Language |
|-----------------------|----------|
| Hello, how are you?  | English  |
| Hola, ¿cómo estás?   | Spanish  |
| Bonjour tout le monde| French   |

- The dataset is sourced from publicly available multilingual datasets, such as:
  - [Language detection dataset](https://www.kaggle.com/datasets/lailaboullous/language-detection-dataset)
  - [dataset](https://www.kaggle.com/datasets/amankumarjha2020/language-detection)
  - Custom curated datasets for additional languages

---

## Model Architecture
- **Preprocessing**: Handles text tokenization, normalization, and encoding.
- **Feature Extraction**: Utilizes n-grams, word embeddings, or TF-IDF vectors.
- **Model**: Implements machine learning and deep learning models (e.g., Logistic Regression, LSTMs, or Transformers) for classification.

---

## Evaluation Metrics
- **Accuracy**: Percentage of correctly identified languages.
- **Precision, Recall, F1-Score**: For multi-class classification.
- **Confusion Matrix**: Visual representation of model performance.

---

## Contributing
We welcome contributions from the community! To contribute:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with a detailed description.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

### Contact
For queries or feedback, reach out to:
- Project Maintainer: SAKTHIVINASH
- Organization: Infosys Internship Program 2024

