"""Build an audio-queryable vector index for multimodal RAG.

For every (filtered) training clip this stores:
  * a Whisper-encoder embedding of the audio      -> Chroma vector
  * the clip itself as 16 kHz mono WAV             -> <db_path>/audio/<collection>/id_N.wav
  * {target, source_text, audio_path, ...}         -> Chroma metadata

At inference time MultiModalRAG embeds the *query* audio with the same
embedder, and GeminiMultimodalGenerator reads the stored WAVs back so the
retrieved examples reach Gemini as audio, not just text.
"""

from pathlib import Path

import chromadb
import datasets
import hydra
from omegaconf import DictConfig

from asr_thesis_project.data.audio_utils import resample, to_mono, write_wav
from asr_thesis_project.data.filters import combined_filter


# config_path is relative to *this file* (src/asr_thesis_project/data/)
@hydra.main(
    version_base=None, config_path="../../configs", config_name="audio_rag_db_generate_config"
)
def main(cfg: DictConfig):
    embedder = hydra.utils.instantiate(cfg.embedding)
    embed_sr = embedder.sampling_rate
    store_sr = cfg.get("audio_sampling_rate", 16000)

    dataset_name = cfg.get("dataset_name", "marsianin500/Speech2Latex")
    split = cfg.get("split", "sentences_train")
    dataset = datasets.load_dataset(dataset_name, name="default", split=split)

    print("Filtering dataset...")
    dataset = dataset.filter(combined_filter, num_proc=cfg.get("num_proc", 10))

    db_path = Path(cfg.get("db_path", "./vector_db"))
    collection_name = cfg.get("collection_name", "speech2latex-whisper-audio")
    audio_rel_dir = Path("audio") / collection_name
    (db_path / audio_rel_dir).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name=collection_name,
        # Recorded so the inference side can refuse to query with a different
        # embedder / sample rate (same-dimension mismatches are otherwise silent).
        metadata={
            "hnsw:space": "cosine",
            "embedder": cfg.embedding._target_,
            "embedder_model": cfg.embedding.get("model_name", ""),
            "embedder_sampling_rate": int(embed_sr),
            "audio_sampling_rate": int(store_sr),
            "source_split": split,
            "has_audio": True,
        },
    )

    total_docs = len(dataset)
    batch_size = cfg.get("batch_size", 32)
    print(f"Indexing {total_docs} clips into '{collection_name}' at {db_path} ...")

    for start in range(0, total_docs, batch_size):
        end = min(total_docs, start + batch_size)
        batch = dataset[start:end]

        source_sentences = batch["whisper_text"]
        target_sentences = batch["sentence"]

        embed_audios, metadatas, ids = [], [], []
        for j, audio in zip(range(start, end), batch["audio_path"]):
            samples = audio.get_all_samples()
            wav = to_mono(samples.data)
            orig_sr = samples.sample_rate

            # 1. waveform for the embedder
            embed_audios.append(resample(wav, orig_sr, embed_sr).numpy())

            # 2. stored clip for the generator (write_wav skips existing files -> resumable)
            doc_id = f"id_{j}"
            rel_path = audio_rel_dir / f"{doc_id}.wav"
            write_wav(db_path / rel_path, resample(wav, orig_sr, store_sr), store_sr)

            ids.append(doc_id)
            metadatas.append(
                {
                    "target": target_sentences[j - start],
                    "source_text": source_sentences[j - start],
                    "audio_path": rel_path.as_posix(),  # relative to db_path
                    "sample_rate": int(store_sr),
                    "duration_s": float(wav.shape[-1] / orig_sr),
                    "dataset_index": int(j),
                }
            )

        embeddings = embedder(embed_audios)

        collection.upsert(
            ids=ids,
            documents=source_sentences,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        if (start // batch_size) % 10 == 0 or end >= total_docs:
            print(f"Indexed {end} / {total_docs}")

    print(f"Done. Collection '{collection_name}' now holds {collection.count()} clips.")


if __name__ == "__main__":
    main()
