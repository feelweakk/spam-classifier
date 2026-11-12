# 📩 Spam Classifier

A machine-learning model that detects spam in SMS messages.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)

## 📊 Results

| Metric | Value |
|---|---|
| Accuracy | **96.68%** |
| Algorithm | Multinomial Naive Bayes |
| Features | TF-IDF |
| Test set | 20% (1,115 messages) |

## 🚀 Usage

```bash
pip install -r requirements.txt

# train the model (creates spam_model.pkl and vectorizer.pkl)
python train.py

# check a message
python predict.py "Congratulations! You won a free iPhone! Click here now!"
# 🚨 SPAM: Congratulations! You won a free iPhone! Click here now!

# interactive mode
python predict.py
```

## ⚙️ How it works

1. **Load data** — 5,572 SMS labelled `ham` (normal) or `spam`
2. **TF-IDF** — each message becomes a vector showing how important each word is
3. **Naive Bayes** — the model learns which words appear more often in spam
4. **Evaluation** — tested on messages the model never saw during training

## 📁 Project structure

```
├── train.py           # training and evaluation
├── predict.py         # check your own messages
├── SMSSpamCollection  # dataset
├── spam_model.pkl     # trained model
└── vectorizer.pkl     # trained TF-IDF vectorizer
```

## 📁 Dataset

[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,572 messages (4,825 ham + 747 spam).

---

Made by **Luka Petrushin** · [fykkkke](https://github.com/feelweakk)
