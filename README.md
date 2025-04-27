# SpamShield - Machine Learning Model

This repository contains the machine learning model training files, datasets, and final serialized models used in the SpamShield spam detection platform.

The model classifies email text into two categories: **Spam** or **Ham** based on natural language features.

---

## Contents

| File | Description |
|:---|:---|
| `email.csv` | Dataset of labeled emails used for training and evaluation |
| `train.model.ipynb`, `train.model1.ipynb` | Jupyter notebooks for training initial and final Naive Bayes models |
| `logistic.regression.ipynb` | Logistic Regression training notebook |
| `random.forest.ipynb` | Random Forest training notebook |
| `spam_classifier.pkl` | Final trained Naive Bayes classifier model |
| `vectorizer.pkl` | Final trained TfidfVectorizer used for feature extraction |
| `requirements.txt` | List of Python packages needed to run training notebooks |

---

## How It Works

- Text data is preprocessed and vectorized using **TF-IDF** (Term Frequency-Inverse Document Frequency).
- A **Multinomial Naive Bayes** classifier is trained on the labeled email dataset.
- Alternative models like **Logistic Regression** and **Random Forest** were explored and compared.
- The final model (`spam_classifier.pkl`) and vectorizer (`vectorizer.pkl`) are serialized and used by the SpamShield backend API for real-time email classification.

---

## Training Workflow

1. Load and preprocess the email dataset (`email.csv`).
2. Vectorize the email text using `TfidfVectorizer`.
3. Train the classification models (Naive Bayes, Logistic Regression, Random Forest).
4. Evaluate models based on accuracy, precision, recall, and F1-score.
5. Select the best performing model for deployment.
6. Save the model and vectorizer using Python's `joblib` library.

---

equirements

Install dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
## Usage

- Open any of the ```.ipynb``` training notebooks.
- Run the training pipeline.

## Integration with SpamShield Backend

- The trained model and vectorizer are loaded by the The trained model and vectorizer are loaded by the **Flask backend**.
- Incoming text is vectorized and classified in real time.
- Predictions and confidence scores are returned to the user interface.
