import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake['label'] = 0   # FAKE
real['label'] = 1   # REAL

# Combine datasets
df = pd.concat([fake, real], axis=0)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Use title + text for better accuracy
df['content'] = df['title'] + " " + df['text']

# Features and labels
X = df['content']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF (optimized for large dataset)
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    max_features=50000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model (increase iterations for convergence)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# 🔍 Prediction function
def predict_news(news_text):
    vec = vectorizer.transform([news_text])
    prediction = model.predict(vec)[0]
    return "REAL" if prediction == 1 else "FAKE"

# Example
sample = "Government announces new economic policy to boost jobs"
print("Prediction:", predict_news(sample))
