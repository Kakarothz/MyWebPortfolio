"""Minimal pretrained Transformer sentiment inference demo."""
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

if __name__ == "__main__":
    samples = [
        "The deployment was smooth and the system is stable.",
        "The application failed repeatedly during the test.",
    ]
    for text, result in zip(samples, classifier(samples)):
        print({"text": text, **result})
