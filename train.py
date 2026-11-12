"""Train a model that detects spam SMS.

Run: python train.py
Saves spam_model.pkl and vectorizer.pkl after training.
"""

import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "message"])

# Turn ham/spam into numbers: ham=0, spam=1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Split into training and test sets (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

# Turn text into numbers with TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate
predictions = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print("\nDetailed report:")
print(classification_report(y_test, predictions, target_names=["ham", "spam"]))

# Save the model and the vectorizer
with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model saved!")
