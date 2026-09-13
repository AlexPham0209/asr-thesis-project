import logging
from typing import Any, Sequence, Union

import chromadb
import torch

from asr_thesis_project.embeddings.embedding import BaseEmbedding
from asr_thesis_project.generator.generator import BaseGenerator, RetrievedExample

logger = logging.getLogger("inference")


class MultiModalRAG:
    """Audio-output RAG with a pluggable retrieval key.

    Two modes, chosen by what you pass as `retrieval_inputs`:
      * audio mode   — retrieve on the query audio itself (Whisper/CLAP embedder,
                       audio index);
      * cascade mode — retrieve on the Whisper transcript of the query
                       (MathBERT embedder, text index).
    In both, the generator receives the query *audio* plus the retrieved
    examples (audio + LaTeX when the index stored clips).
    """

    def __init__(
        self,
        system_prompt: str,
        generator: BaseGenerator,
        embedding: BaseEmbedding,
        collection: chromadb.Collection,
    ):
        self.system_prompt = system_prompt
        self.generator = generator
        self.embedding = embedding
        self.collection = collection

    def retrieve(
        self, queries: Sequence[Any], top_n: int = 3
    ) -> list[list[RetrievedExample]]:
        """`queries` must be whatever the configured embedder accepts:
        waveform tensors for WhisperEmbedding/CLAPEmbedding, strings for MathBERTEmbedding."""
        if top_n <= 0:
            return [[] for _ in queries]

        # Never ask Chroma for more neighbours than the index holds; it warns
        # and truncates in some versions and raises in others.
        available = self.collection.count()
        n_results = min(top_n, available)
        if n_results == 0:
            logger.warning(
                f"Collection '{self.collection.name}' is empty; generating without examples."
            )
            return [[] for _ in queries]

        embeddings_list = self.embedding(list(queries))

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

    def inference_with_examples(
        self,
        inputs: Sequence[torch.Tensor],
        top_n: int = 3,
        retrieval_inputs: Sequence[Any] | None = None,
        hints: Sequence[str] | None = None,
    ) -> tuple[list[str], list[list[RetrievedExample]]]:
        """Retrieve on `retrieval_inputs` (default: `inputs`), generate on `inputs`.

        `hints` are optional per-input strings (e.g. the Whisper transcript) the
        generator may show the model alongside the audio. Returns the predictions
        and the examples used for each, so callers can log the neighbours.
        """
        queries = list(inputs)
        keys = list(retrieval_inputs) if retrieval_inputs is not None else queries
        if len(keys) != len(queries):
            raise ValueError(
                f"retrieval_inputs ({len(keys)}) and inputs ({len(queries)}) differ in length"
            )

        batched_examples = self.retrieve(keys, top_n)
        predictions = self.generator.generate(
            inputs=queries, batched_examples=batched_examples, hints=hints
        )
        return predictions, batched_examples

    def inference(
        self,
        inputs: Union[torch.Tensor, Sequence[torch.Tensor]],
        top_n: int = 3,
        retrieval_inputs: Sequence[Any] | None = None,
        hints: Sequence[str] | None = None,
    ) -> Union[str, list[str]]:
        is_single = isinstance(inputs, torch.Tensor)
        queries = [inputs] if is_single else list(inputs)
        if is_single and retrieval_inputs is not None and not isinstance(retrieval_inputs, list):
            retrieval_inputs = [retrieval_inputs]
        if is_single and isinstance(hints, str):
            hints = [hints]

        predictions, _ = self.inference_with_examples(
            queries, top_n=top_n, retrieval_inputs=retrieval_inputs, hints=hints
        )
        return predictions[0] if is_single else predictions
