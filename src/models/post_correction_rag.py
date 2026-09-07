from abc import ABC, abstractmethod
import asyncio
import logging
import os
import warnings
from typing import Union

import chromadb
from google import genai
from google.genai import types
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import nest_asyncio

from generator.generator import BaseGenerator

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("inference")


class PostCorrectionRAG:
    # Passed the ChromaDB collection into the initialization
    def __init__(
        self,
        system_prompt: str,
        generator: BaseGenerator,
        collection: chromadb.Collection,
    ):
        self.system_prompt = system_prompt
        self.generator = generator
        self.collection = collection

    def retrieve(
        self, inputs: Union[list[str], str], top_n: int = 3
    ) -> list[list[str]]:
        # Standardize inputs to list and assign to queries
        queries = [inputs] if isinstance(inputs, str) else inputs

        results = self.collection.query(
            query_texts=queries,
            n_results=top_n,
            include=["documents", "metadatas", "distances"],
        )

        batch_examples = []
        for i in range(len(queries)):
            source_sentences = results["documents"][i]
            metadatas = results["metadatas"][i]

            examples = []
            for source, meta in zip(source_sentences or [], metadatas or []):
                target = (meta or {}).get("target", "") if meta else ""
                s = f"Original sentence: {source}\nLaTeX corrected sentence: {target}"
                examples.append(s)

            batch_examples.append(examples)

        return batch_examples

    def inference(self, inputs: Union[list[str], str]) -> Union[list[str], str]:
        is_single = isinstance(inputs, str)
        queries = [inputs] if is_single else inputs

        batched_examples = self.retrieve(queries)
        res = self.generator.generate(inputs=queries, batched_examples=batched_examples)

        return res[0] if is_single else res
