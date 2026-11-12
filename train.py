"""Обучение модели для определения спама в SMS.

Запуск: python train.py
После обучения сохраняет spam_model.pkl и vectorizer.pkl
"""

import pickle
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

sys.stdout.reconfigure(encoding="utf-8")  # кириллица в консоли Windows

# Загружаем датасет
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "message"])

# Превращаем ham/spam в числа: ham=0, spam=1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Делим на тренировочную и тестовую выборки (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

# Превращаем текст в числа через TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Обучаем модель
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Проверяем точность
predictions = model.predict(X_test_tfidf)
print(f"Точность: {accuracy_score(y_test, predictions):.2%}")
print("\nПодробный отчёт:")
print(classification_report(y_test, predictions, target_names=["ham", "spam"]))

# Сохраняем модель и векторизатор
with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Модель сохранена!")
