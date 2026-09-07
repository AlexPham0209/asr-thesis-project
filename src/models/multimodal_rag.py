from typing import Any, List, Union
import chromadb
import torch

from embeddings.embedding import BaseEmbedding
from generator.generator import BaseGenerator


class MultiModalRAG:
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
        self, inputs: Union[str, torch.Tensor, List[Any]], top_n: int = 3
    ) -> List[List[str]]:
        # Normalize single inputs into a list
        if not isinstance(inputs, list):
            inputs = [inputs]

        # Convert PyTorch Tensors to list format if necessary
        formatted_inputs = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                formatted_inputs.append(inp.detach().cpu().numpy().tolist())
            else:
                formatted_inputs.append(inp)

        embeddings_list = self.embedding(formatted_inputs)

        results = self.collection.query(
            query_embeddings=embeddings_list, n_results=top_n
        )

        batch_examples = []

        for i in range(len(embeddings_list)):
            source_sentences = results["documents"][i]
            metadatas = results["metadatas"][i]

            examples = []
            for source, meta in zip(source_sentences, metadatas):
                target = meta.get("target", "") if meta else ""
                s = f"Original sentence: {source}\nLaTeX corrected sentence: {target}"
                examples.append(s)

            batch_examples.append(examples)

        return batch_examples

    def inference(self, inputs: Union[str, torch.Tensor, List[Any]]) -> Union[str, List[str]]:
        # FIXED: Correctly detect single inputs whether str or torch.Tensor
        is_single = not isinstance(inputs, list)
        queries = [inputs] if is_single else inputs

        batched_examples = self.retrieve(queries)
        res = self.generator.generate(inputs=queries, batched_examples=batched_examples)

        return res[0] if is_single else res