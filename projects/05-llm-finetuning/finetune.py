"""Educational parameter-efficient fine-tuning template.

No pretrained checkpoint or performance result is claimed by this repository.
Provide your own dataset with 'text' field before running.
"""
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

p = argparse.ArgumentParser()
p.add_argument("--model", default="distilgpt2")
p.add_argument("--data", required=True, help="JSONL file containing a text field")
args = p.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model)
ds = load_dataset("json", data_files=args.data)["train"]
def encode(batch):
    enc = tok(batch["text"], truncation=True, padding="max_length", max_length=128)
    enc["labels"] = [x[:] for x in enc["input_ids"]]
    return enc
encoded = ds.map(encode, batched=True, remove_columns=ds.column_names)
training = TrainingArguments(output_dir="./artifacts", num_train_epochs=1, per_device_train_batch_size=2, save_strategy="no", report_to="none")
Trainer(model=model, args=training, train_dataset=encoded).train()
model.save_pretrained("./artifacts/model"); tok.save_pretrained("./artifacts/model")
