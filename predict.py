"""Проверка сообщений обученной моделью.

Запуск:
    python predict.py "Congratulations! You won a free iPhone!"
    python predict.py            # интерактивный режим
"""

import pickle
import sys

# Чтобы эмодзи и кириллица нормально печатались в консоли Windows
sys.stdout.reconfigure(encoding="utf-8")

with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def check_message(text: str) -> bool:
    """Возвращает True, если сообщение — спам."""
    return model.predict(vectorizer.transform([text]))[0] == 1


def print_result(text: str):
    if check_message(text):
        print(f"🚨 СПАМ: {text}")
    else:
        print(f"✅ НЕ СПАМ: {text}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print_result(" ".join(sys.argv[1:]))
    else:
        print("Введи сообщение (пустая строка — выход):")
        while text := input("> ").strip():
            print_result(text)
