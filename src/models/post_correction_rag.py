from abc import ABC, abstractmethod
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

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("inference")

class BaseGenerator(ABC):
    @abstractmethod
    def generate(
        self, inputs: list[str], batched_examples: list[list[str]]
    ) -> list[str]:
        pass
    

class GeminiGenerator(BaseGenerator):
    def __init__(self, system_prompt: str, model_name: str = "gemini-2.5-flash"):
        self.system_prompt = system_prompt
        self.client = genai.Client()
        self.model_name = model_name

    def generate(self, inputs: list[str], batched_examples: list[list[str]]) -> list[str]:
        responses = []
        # Note: For production with large batches, consider ThreadPoolExecutor here
        for input_text, examples in zip(inputs, batched_examples):
            examples_str = "\n\n".join(examples)
            dynamic_system_prompt = (
                f"{self.system_prompt}\n\n<examples>\n{examples_str}\n</examples>"
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=input_text,
                config=types.GenerateContentConfig(
                    system_instruction=dynamic_system_prompt,
                    temperature=0.2,
                    max_output_tokens=256,
                ),
            )
            responses.append(response.text)
        return responses
    
    
class HuggingFaceGenerator(BaseGenerator):
    def __init__(self, system_prompt: str, model_path: str, tokenizer_path: str, peft_path: str | None = None):
        super().__init__()
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.peft_path = peft_path
        self.system_prompt = system_prompt
        
        logger.info(f"Loading tokenizer: {self.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loading LLM model: {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        
        if self.peft_path and os.path.exists(self.peft_path):
            logger.info(f"Applying LoRA weights from: {self.peft_path}")
            self.model = PeftModel.from_pretrained(self.model, self.peft_path)
        
        self.model.eval()
        
    def generate(self, inputs: list[str], batched_examples: list[list[str]]) -> list[str]:
        prompts = []
        for i, input_text in enumerate(inputs):
            examples_str = "\n\n".join(batched_examples[i])
            system_prompt = (
                f"{self.system_prompt}\n\n<examples>\n{examples_str}\n</examples>"
            )
        
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
            ]
        
            prompt = self.tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
        
        encoded_inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **encoded_inputs,
                max_new_tokens=256,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=0.2,
                do_sample=True,
            )
        
        prompt_length = encoded_inputs.input_ids.shape[-1]
        generated_ids = outputs[:, prompt_length:]
        
        # Decode the whole batch
        res = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        return res
    

class PostCorrectionRAG:
    # Passed the ChromaDB collection into the initialization
    def __init__(self, system_prompt: str, generator: BaseGenerator, collection: chromadb.Collection):
        self.system_prompt = system_prompt
        self.generator = generator
        self.collection = collection

    def retrieve(self, inputs: Union[list[str], str], top_n: int = 3) -> list[list[str]]:
        # Standardize inputs to list and assign to queries
        queries = [inputs] if isinstance(inputs, str) else inputs

        results = self.collection.query(query_texts=queries, n_results=top_n)

        batch_examples = []
        for i in range(len(queries)):
            source_sentences = results["documents"][i]
            metadatas = results["metadatas"][i]

            examples = []
            for source, meta in zip(source_sentences, metadatas):
                target = meta.get("target", "") if meta else ""
                s = f"Original sentence: {source}\nLaTeX corrected sentence: {target}"
                examples.append(s)

            batch_examples.append(examples)

        return batch_examples

    def inference(self, inputs: Union[list[str], str]) -> Union[list[str], str]:
        is_single = isinstance(inputs, str)
        queries = [inputs] if is_single else inputs

        batched_examples = self.retrieve(queries)
        res = self.generator.generate(
            inputs=queries, 
            batched_examples=batched_examples
        )
        
        return res[0] if is_single else res