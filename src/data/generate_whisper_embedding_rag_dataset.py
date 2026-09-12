import chromadb
import datasets
import torchaudio
from filters import combined_filter
from embeddings.embedding import WhisperEmbedding


def main():
    # 1. Initialize wrapper class (handles model loading & device selection automatically)
    embedder = WhisperEmbedding(model_name="openai/whisper-small", sampling_rate=16000)

    # 2. Load dataset and match sampling rate with the embedder
    dataset = datasets.load_dataset(
        "marsianin500/Speech2Latex", name="default", split="sentences_train"
    )
    dataset = dataset.cast_column(
        "audio", datasets.Audio(sampling_rate=embedder.sampling_rate)
    )

    print("Filtering dataset...")
    dataset = dataset.filter(combined_filter, num_proc=10)

    # 3. Configure ChromaDB
    client = chromadb.PersistentClient(path="./vector_db")
    collection = client.get_or_create_collection(
        name="speech2latex-Whisper", metadata={"hnsw:space": "cosine"}
    )

    total_docs = len(dataset)
    GPU_BATCH_SIZE = 16
    print(f"Generating embeddings and upserting {total_docs} documents...")

    # 4. Stream and upsert
    for start in range(0, total_docs, GPU_BATCH_SIZE):
        end = min(total_docs, start + GPU_BATCH_SIZE)
        batch = dataset[start:end]

        source_sentences = batch["whisper_text"]
        target_sentences = batch["sentence"]

        # Extract audio waveform arrays
        audios = []
        for audio in batch["audio_path"]:
            samples = audio.get_all_samples()
            audio_tensor = samples.data.squeeze(dim=0)

            if samples.sample_rate != embedder.sampling_rate:
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor,
                    orig_freq=samples.sample_rate,
                    new_freq=embedder.sampling_rate,
                )
            audios.append(audio_tensor.numpy())

        # 5. Generate embeddings via __call__
        embeddings = embedder(audios)

        # Prepare payload
        batch_ids = [f"id_{j}" for j in range(start, end)]
        batch_meta = [{"target": t} for t in target_sentences]

        collection.upsert(
            documents=source_sentences,
            embeddings=embeddings,
            metadatas=batch_meta,
            ids=batch_ids,
        )

        if (start // GPU_BATCH_SIZE) % 10 == 0 or end >= total_docs:
            print(f"Processed & Upserted {end} / {total_docs} items...")

    print("Vector database created successfully.")


if __name__ == "__main__":
    main()
