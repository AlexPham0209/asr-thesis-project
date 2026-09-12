import chromadb
import datasets

from filters import combined_filter


def main():
    # Filtering dataset
    dataset = datasets.load_dataset(
        "marsianin500/Speech2Latex", name="default", split="sentences_train"
    )
    dataset = dataset.filter(combined_filter, num_proc=10)

    client = chromadb.PersistentClient(path="./vector_db")
    collection = client.get_or_create_collection(name="speech2latex-text")

    source_sentences = dataset["whisper_text"]
    target_sentences = [{"target": target} for target in dataset["sentence"]]
    ids = [f"id_{i}" for i in range(len(dataset))]

    BATCH_SIZE = 5000
    print(f"Upserting {len(ids)} documents in batches of {BATCH_SIZE}...")
    for start in range(0, len(ids), BATCH_SIZE):
        end = min(len(ids), start + BATCH_SIZE)
        batch_docs = source_sentences[start:end]
        batch_meta = target_sentences[start:end]
        batch_ids = ids[start:end]

        collection.upsert(documents=batch_docs, metadatas=batch_meta, ids=batch_ids)
        print(f"Upserted batch {start // BATCH_SIZE + 1}")

    print("Vector database created")


if __name__ == "__main__":
    main()
