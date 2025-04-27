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

## Model Evaluation

Several classification models were trained and evaluated to determine the best-performing approach for email spam detection.

The evaluation focused on metrics such as accuracy, precision, recall, and F1-score.  
The final selected model needed to balance high accuracy with strong generalization to unseen email examples.

The following models were tested:

| Model | Accuracy | Precision (Macro Avg) | Recall (Macro Avg) | F1-Score (Macro Avg) |
|:---|:---|:---|:---|:---|
| Naive Bayes | **98.0%** | 98% | 98% | **98%** |
| Random Forest Classifier | 98.3% | 99% | 94% | 96% |
| Logistic Regression | 96.2% | 98% | 86% | 91% |

---

### Model Selection Rationale

While the Random Forest model slightly outperformed Naive Bayes on accuracy, it showed lower recall and added significantly more complexity and training overhead.  
Logistic Regression also exhibited lower recall for spam detection, which is critical in this use case.

**Naive Bayes** was ultimately selected for SpamShield because:
- It achieved a strong balance across all evaluation metrics.
- It offers lightweight, fast predictions — ideal for real-time classification needs.
- It generalizes well without overfitting, making it reliable across varied email inputs.

Thus, Naive Bayes provided the best overall trade-off between **accuracy, efficiency, and scalability** for real-time deployment.

## Requirements

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
