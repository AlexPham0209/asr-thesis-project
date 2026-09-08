import chromadb
import datasets
import torch
import functools
from transformers import AutoModel, AutoTokenizer

from data.filters import combined_filter


def preprocess(batch, model, tokenizer):
    source_sentences = batch["whisper_text"]
    target_sentences = batch["sentence"]

    inputs = tokenizer(
        source_sentences,
        padding=True,
        truncation=True,
        max_length=250,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs).cpu().tolist()

    return {
        "sources": source_sentences,
        "targets": target_sentences,
        "embeddings": text_features,
    }


def main():
    model = AutoModel.from_pretrained("laion/clap-htsat-unfused", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

    dataset = datasets.load_dataset(
        "marsianin500/Speech2Latex", name="default", split="sentences_train"
    )

    print("Filtering dataset...")
    dataset = dataset.filter(combined_filter, num_proc=10)

    preprocess_fn = functools.partial(
        preprocess,
        model=model,
        tokenizer=tokenizer,
    )

    print("Generating embeddings...")
    dataset = dataset.map(
        preprocess_fn,
        batched=True,
        batch_size=16,
        remove_columns=dataset.column_names,  # OPTIMIZATION: Discard old columns to save disk space
    )

    client = chromadb.PersistentClient(path="./vector_db")
    collection = client.get_or_create_collection(name="speech2latex-CLAP")

    BATCH_SIZE = 5000
    total_docs = len(dataset)
    print(f"Upserting {total_docs} documents in batches of {BATCH_SIZE}...")

    for i in range(0, total_docs, BATCH_SIZE):
        batch = dataset[i : i + BATCH_SIZE]

        batch_docs = batch["sources"]
        batch_meta = [{"target": t} for t in batch["targets"]]
        batch_embeddings = batch["embeddings"]

        # Generate IDs dynamically for the current batch
        batch_ids = [f"id_{j}" for j in range(i, i + len(batch_docs))]

        collection.upsert(
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_meta,
            ids=batch_ids,
        )
        print(
            f"Upserted batch {i // BATCH_SIZE + 1} / {(total_docs // BATCH_SIZE) + 1}"
        )

    print("Vector database created successfully.")


if __name__ == "__main__":
    main()
