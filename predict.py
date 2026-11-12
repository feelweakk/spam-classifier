"""Check messages with the trained model.

Run:
    python predict.py "Congratulations! You won a free iPhone!"
    python predict.py            # interactive mode
"""

import pickle
import sys

# Make emoji print correctly in the Windows console
sys.stdout.reconfigure(encoding="utf-8")

with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def check_message(text: str) -> bool:
    """Returns True if the message is spam."""
    return model.predict(vectorizer.transform([text]))[0] == 1


def print_result(text: str):
    if check_message(text):
        print(f"🚨 SPAM: {text}")
    else:
        print(f"✅ NOT SPAM: {text}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print_result(" ".join(sys.argv[1:]))
    else:
        print("Type a message (empty line to quit):")
        while text := input("> ").strip():
            print_result(text)
