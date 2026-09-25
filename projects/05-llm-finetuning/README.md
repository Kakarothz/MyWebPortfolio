# 05 — LLM Fine-Tuning Lab

**Status: Implemented template**

A transparent educational fine-tuning workflow using Hugging Face `Trainer`. The script accepts a JSONL dataset containing a `text` field and a configurable causal-language-model checkpoint.

## Run
```bash
pip install -r requirements.txt
python finetune.py --data your-data.jsonl --model distilgpt2
```

No dataset, trained checkpoint or performance result is committed or claimed. Users must supply appropriate data and review licensing, privacy and compute requirements before training.
