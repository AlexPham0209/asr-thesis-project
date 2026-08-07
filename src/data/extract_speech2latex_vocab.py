import json
import os

from datasets import load_dataset

HF_TOKEN = os.environ['HF_TOKEN']

processed_path = os.path.join("data", "processed")
dataset_path = os.path.join(processed_path, "speech2latex")
os.makedirs(dataset_path, exist_ok=True)

dataset = load_dataset("marsianin500/Speech2Latex", split="sentences_train+sentences_test", token=HF_TOKEN, streaming=True) 
dataset = dataset.with_format(None)
dataset = dataset.remove_columns(["audio_path"])
dataset = dataset.filter(lambda sample: sample["language"] == "eng")

characters = set()

for sample in dataset.take(2000):
    text = sample["sentence"]
    characters.update(text)

vocab = {i:character for i, character in enumerate(sorted(list(characters)))}
vocab["[UNK]"] = len(vocab)
vocab["[PAD]"] = len(vocab)

print(vocab)
with open(os.path.join(dataset_path, "vocab.json"), "w") as file:
    json.dump(vocab, file)