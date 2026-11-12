# 📩 Spam Classifier

Модель машинного обучения, которая определяет спам в SMS-сообщениях.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)

## 📊 Результаты

| Метрика | Значение |
|---|---|
| Точность (accuracy) | **96.68%** |
| Алгоритм | Multinomial Naive Bayes |
| Признаки | TF-IDF |
| Тестовая выборка | 20% (1115 сообщений) |

## 🚀 Запуск

```bash
pip install -r requirements.txt

# обучить модель (создаёт spam_model.pkl и vectorizer.pkl)
python train.py

# проверить сообщение
python predict.py "Congratulations! You won a free iPhone! Click here now!"
# 🚨 СПАМ: Congratulations! You won a free iPhone! Click here now!

# интерактивный режим
python predict.py
```

## ⚙️ Как это работает

1. **Загрузка данных** — 5572 SMS с метками `ham` (обычное) и `spam`
2. **TF-IDF** — каждое сообщение превращается в вектор: насколько важно каждое слово
3. **Naive Bayes** — модель учится, какие слова чаще встречаются в спаме
4. **Оценка** — проверка на сообщениях, которые модель не видела при обучении

## 📁 Структура

```
├── train.py           # обучение и оценка модели
├── predict.py         # проверка своих сообщений
├── SMSSpamCollection  # датасет
├── spam_model.pkl     # обученная модель
└── vectorizer.pkl     # обученный TF-IDF векторизатор
```

## 📁 Датасет

[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5572 сообщения (4825 ham + 747 spam).

---

Автор: **Лука Петрушин** · [fykkkke](https://github.com/feelweakk)
