import chromadb
import datasets

def combined_filter(sample):
    # Language check
    if sample["language"] != "eng":
        return False

    # Equation quality checks
    text = sample["whisper_text"]

    audio_data = sample["audio_path"].get_all_samples().data
    if not (audio_data.ndim == 2 and audio_data.shape[0] == 1):
        return False

    return True


def main():
    # Filtering dataset
    dataset = datasets.load_dataset(
        "marsianin500/Speech2Latex", name="default", split="sentences_train"
    )
    dataset = dataset.filter(combined_filter, num_proc=10)

    client = chromadb.PersistentClient(path="./vector_db")
    collection = client.get_or_create_collection(name="speech2latex")

    source_sentences = dataset["whisper_text"]
    target_sentences = [{"target": target} for target in dataset["sentence"]]
    ids = [f"id_{i}" for i in range(len(dataset))]

    BATCH_SIZE = 5000 
    print(f"Upserting {len(ids)} documents in batches of {BATCH_SIZE}...")
    for i in range(0, len(ids), BATCH_SIZE):
        batch_docs = source_sentences[i : i + BATCH_SIZE]
        batch_meta = target_sentences[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]
            
        collection.upsert(
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids
        )
        print(f"Upserted batch {i // BATCH_SIZE + 1}")
    
    print("Vector database created")


if __name__ == "__main__":
    main()