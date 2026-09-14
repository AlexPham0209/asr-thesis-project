"""Build a text-queryable vector index for post-correction / cascade RAG.

For every (filtered) training row this stores:
  * a MathBERT embedding of `whisper_text` (the ASR output)  -> Chroma vector
  * `whisper_text`                                            -> Chroma document
  * {target: sentence, ...}                                   -> Chroma metadata
  * (store_audio) the clip as 16 kHz mono WAV                 -> <db_path>/audio/<collection>/id_N.flac
                                                                 + `audio_path` in the metadata

Queried by PostCorrectionRAG (text in, text out) and by the cascade mode of
multimodal_rag_inference.py, where the query is the Whisper transcript of a
test clip and Gemini receives the retrieved examples as audio + LaTeX.
"""

from pathlib import Path

import chromadb
import datasets
import hydra
from omegaconf import DictConfig

from asr_thesis_project.utils.audio import decode_audio, write
from asr_thesis_project.data.filters import combined_filter


# config_path is relative to *this file* (src/asr_thesis_project/data/)
@hydra.main(
    version_base=None, config_path="../../configs", config_name="text_rag_db_generate_config"
)
def main(cfg: DictConfig):
    embedder = hydra.utils.instantiate(cfg.embedding)

    dataset_name = cfg.get("dataset_name", "marsianin500/Speech2Latex")
    split = cfg.get("split", "sentences_train")
    dataset = datasets.load_dataset(dataset_name, name="default", split=split)

    print("Filtering dataset...")
    dataset = dataset.filter(combined_filter, num_proc=cfg.get("num_proc", 10))

    db_path = Path(cfg.get("db_path", "./vector_db"))
    collection_name = cfg.get("collection_name", "speech2latex-mathbert-text")
    store_audio = cfg.get("store_audio", True)
    audio_sampling_rate = cfg.get("audio_sampling_rate", 16000)
    audio_rel_dir = Path("audio") / collection_name

    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "embedder": cfg.embedding._target_,
            "embedder_model": cfg.embedding.get("model_name", ""),
            "source_split": split,
            "has_audio": bool(store_audio),
            "audio_sampling_rate": int(audio_sampling_rate),
        },
    )

    total_docs = len(dataset)
    batch_size = cfg.get("batch_size", 32)
    print(
        f"Indexing {total_docs} sentences into '{collection_name}' at {db_path} "
        f"(store_audio={store_audio}) ..."
    )

    for start in range(0, total_docs, batch_size):
        end = min(total_docs, start + batch_size)
        batch = dataset[start:end]

        source_sentences = batch["whisper_text"]
        target_sentences = batch["sentence"]

        ids, metadatas = [], []
        for k, j in enumerate(range(start, end)):
            doc_id = f"id_{j}"
            meta = {"target": target_sentences[k], "dataset_index": int(j)}

            if store_audio:
                rel_path = audio_rel_dir / f"{doc_id}.flac"
                
                out_path = db_path / rel_path
                if out_path.exists():
                    duration = None
                else:
                    wav, duration = decode_audio(batch["audio_path"][k], audio_sampling_rate)
                    write(out_path, wav, audio_sampling_rate)
                
                meta["audio_path"] = rel_path.as_posix() 
                meta["sample_rate"] = int(audio_sampling_rate)
                if duration is not None:
                    meta["duration"] = float(duration)

            ids.append(doc_id)
            metadatas.append(meta)

        embeddings = embedder(source_sentences)

        collection.upsert(
            ids=ids,
            documents=source_sentences,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        if (start // batch_size) % 10 == 0 or end >= total_docs:
            print(f"Indexed {end} / {total_docs}")

    print(f"Done. Collection '{collection_name}' now holds {collection.count()} sentences.")


if __name__ == "__main__":
    main()
