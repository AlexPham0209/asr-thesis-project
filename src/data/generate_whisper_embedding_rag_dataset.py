import functools
import chromadb
import datasets
import torch
import torch.nn.functional as F
from transformers import WhisperProcessor

from data.filters import combined_filter
from models.clap_model import WhisperEmbedding


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Use proper CLAP model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    embedding = WhisperEmbedding(
        model_name="openai/whisper-large-v3"
    )
    model.eval()  # Disable dropout/batchnorm updates

    # 2. Load and filter dataset
    dataset = datasets.load_dataset(
        "marsianin500/Speech2Latex", name="default", split="sentences_train"
    )

    print("Filtering dataset...")
    dataset = dataset.filter(combined_filter, num_proc=10)

    # 3. Configure ChromaDB with Cosine Distance
    client = chromadb.PersistentClient(path="./vector_db")
    collection = client.get_or_create_collection(
        name="speech2latex-CLAP",
        metadata={"hnsw:space": "cosine"}  # Set metric to cosine similarity
    )

    total_docs = len(dataset)
    GPU_BATCH_SIZE = 256  # Efficient batch size for GPU inference & Chroma upserts
    print(f"Generating embeddings and upserting {total_docs} documents...")

    # 4. Direct streaming inference and upserting (No map caching overhead)
    for i in range(0, total_docs, GPU_BATCH_SIZE):
        batch = dataset[i : i + GPU_BATCH_SIZE]

        source_sentences = batch["whisper_text"]
        target_sentences = batch["sentence"]

        # Tokenize using ClapProcessor
        inputs = processor(
            text=source_sentences,
            padding=True,
            truncation=True,
            max_length=250,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            # Normalize embeddings for Cosine distance
            text_features = F.normalize(text_features, p=2, dim=-1)
            embeddings = text_features.cpu().tolist()

        # Prepare payload
        batch_ids = [f"id_{j}" for j in range(i, i + len(source_sentences))]
        batch_meta = [{"target": t} for t in target_sentences]

        # Upsert directly to Chroma
        collection.upsert(
            documents=source_sentences,
            embeddings=embeddings,
            metadatas=batch_meta,
            ids=batch_ids,
        )

        if (i // GPU_BATCH_SIZE) % 10 == 0 or (i + GPU_BATCH_SIZE) >= total_docs:
            processed = min(i + GPU_BATCH_SIZE, total_docs)
            print(f"Processed & Upserted {processed} / {total_docs} items...")

    print("Vector database created successfully.")


if __name__ == "__main__":
    main()