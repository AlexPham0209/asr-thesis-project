import logging
from typing import Union

import chromadb

from asr_thesis_project.embeddings.embedding import BaseEmbedding
from asr_thesis_project.generator.generator import BaseGenerator, RetrievedExample

logger = logging.getLogger("inference")


class PostCorrectionRAG:
    """Text-query RAG: embed the raw ASR sentence, pull the nearest training
    sentences (ASR text -> LaTeX target), hand them to the generator as few-shot."""

    def __init__(
        self,
        system_prompt: str,
        collection: chromadb.Collection,
        generator: BaseGenerator,
        embedding: BaseEmbedding,
    ):
        if embedding is None:
            # Querying with query_texts would make Chroma embed with its default
            # MiniLM model — a different space from whatever built the index.
            raise ValueError("PostCorrectionRAG requires the embedder that built the index.")
        self.system_prompt = system_prompt
        self.collection = collection
        self.generator = generator
        self.embedding = embedding

    def retrieve(
        self, inputs: Union[list[str], str], top_n: int = 3
    ) -> list[list[RetrievedExample]]:
        queries = [inputs] if isinstance(inputs, str) else list(inputs)

        if top_n <= 0:
            return [[] for _ in queries]

        available = self.collection.count()
        n_results = min(top_n, available)
        if n_results == 0:
            logger.warning(
                f"Collection '{self.collection.name}' is empty; generating without examples."
            )
            return [[] for _ in queries]

        embeddings_list = self.embedding(queries)

        results = self.collection.query(
            query_embeddings=embeddings_list,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        batch_examples: list[list[RetrievedExample]] = []
        for ids, docs, metas, dists in zip(
            results["ids"], results["documents"], results["metadatas"], results["distances"]
        ):
            examples = []
            for id, doc, meta, dist in zip(ids, docs, metas, dists):
                meta = dict(meta or {})
                examples.append(
                    RetrievedExample(
                        id=id,
                        document=doc or "",
                        target=str(meta.get("target", "")),
                        distance=dist,
                        audio_path=meta.get("audio_path"),
                        metadata=meta,
                    )
                )
            batch_examples.append(examples)

        return batch_examples

    def inference(
        self, inputs: Union[list[str], str], top_n: int = 3
    ) -> Union[list[str], str]:
        is_single = isinstance(inputs, str)
        queries = [inputs] if is_single else list(inputs)

        batched_examples = self.retrieve(queries, top_n)
        res = self.generator.generate(inputs=queries, batched_examples=batched_examples)

        return res[0] if is_single else res
