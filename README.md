# Fake News Detection

This project is a Machine Learning-based Fake News Detection system that uses natural language processing (NLP) techniques to identify and classify news articles as either genuine or fake. The goal of the project is to combat misinformation by leveraging predictive models.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Model](#model)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

---

## Introduction
The spread of fake news has become a significant concern in the digital age, impacting societies and democracies worldwide. This project focuses on building a reliable machine learning model to detect fake news articles by analyzing their text.

---

## Features
- Text preprocessing with techniques like tokenization, stemming, and vectorization (TF-IDF).
- Binary classification (Fake vs. Real news).
- Integration of machine learning models such as Logistic Regression, Naïve Bayes, and Random Forest.
- Evaluation metrics: Accuracy, Precision, Recall, and F1-Score.
- Easy-to-use command-line interface for predictions.

---

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/fake-news-detection.git
    cd fake-news-detection
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1. **Run the App for training and processing the datatset**
    ```bash
    python app.py
    ```
2. **Make Predictions**
    ```bash
    python predict.py --input "Your news article text here"
    ```

---

## Dataset

The dataset used for this project contains labeled articles as "Fake" or "Real". Example datasets include:
- [Kaggle: Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- [LIAR Dataset](https://www.cs.pitt.edu/~chou/data/)

Ensure the dataset is in the correct format with `text` and `label` columns.

---

## Model

### Algorithms Used:
- Logistic Regression
- Naïve Bayes
- Random Forest Classifier

### Text Vectorization:
- Term Frequency-Inverse Document Frequency (TF-IDF)

### Evaluation:
The model performance is measured using standard metrics such as accuracy, precision, recall, and F1-score.

---

## Results

| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.95     | 0.94      | 0.96   | 0.95     |
| Naïve Bayes         | 0.92     | 0.91      | 0.93   | 0.92     |
| Random Forest       | 0.96     | 0.95      | 0.97   | 0.96     |

---

## Contributing

Contributions are welcome! If you would like to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [Scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)
- [Pandas](https://pandas.pydata.org/)
- [Kaggle Datasets](https://www.kaggle.com/)

---

