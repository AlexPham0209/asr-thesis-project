import chromadb
import torch

class PostCorrectionRAG:
    def __init__(self, system_prompt, model, tokenizer, db_path, collection_name):
        self.system_prompt = system_prompt
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.tokenizer = tokenizer
        self.model = model
        
        # Ensure tokenizer has a pad token for batched inference
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def retrieve(self, queries: list | str, top_n=3):
        # 1. Standardize input to a list
        if isinstance(queries, str):
            queries = [queries]

        # 2. Chroma uses 'n_results', not 'top_n'
        results = self.collection.query(
            query_texts=queries,
            n_results=top_n
        )
        
        batch_examples = []
        
        # 3. Iterate through the batch. Chroma returns a 2D list for documents/metadatas.
        for i in range(len(queries)):
            source_sentences = results["documents"][i]
            metadatas = results["metadatas"][i]
            
            examples = []
            for source, meta in zip(source_sentences, metadatas):
                target = meta.get("target", "") if meta else ""
                s = (
                    f"Original sentence: {source}\n"
                    f"LaTeX corrected sentence: {target}"
                )
                examples.append(s)
                
            batch_examples.append(examples)
            
        # Returns List[List[str]] containing examples for each query
        return batch_examples
        
    def inference(self, inputs: list | str):
        # Standardize input to a list
        is_single = isinstance(inputs, str)
        if is_single:
            inputs = [inputs]

        # Get batched examples: List of example lists corresponding to each input
        batched_examples = self.retrieve(inputs)
        
        prompts = []
        
        # 4. Build a unique prompt for each item in the batch
        for i, input_text in enumerate(inputs):
            examples_str = "\n\n".join(batched_examples[i])
            system_prompt = f"{self.system_prompt}\n\n<examples>\n{examples_str}\n</examples>"

            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
            ]
            
            prompt = self.tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        # 5. Tokenize the entire batch at once
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

        # 6. Slice the output dynamically based on the batch's prompt length
        prompt_length = encoded_inputs.input_ids.shape[-1]
        generated_ids = outputs[:, prompt_length:]
        
        # Decode the whole batch
        res = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Return a string if a single string was passed, otherwise return the list
        return res[0] if is_single else res