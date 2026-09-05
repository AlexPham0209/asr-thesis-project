import chromadb
import torch
import torchaudio.functional as F


class MultimodalRAG:
    def __init__(
        self, system_prompt, llm_model, llm_processor, db_path, collection_name
    ):
        self.system_prompt = system_prompt
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        self.llm_processor = llm_processor
        self.llm_processor.tokenizer.padding_side = "left"

        if self.llm_processor.tokenizer.pad_token is None:
            self.llm_processor.tokenizer.pad_token = (
                self.llm_processor.tokenizer.eos_token
            )

        self.llm_model = llm_model
        self.target_sampling_rate = self.llm_processor.feature_extractor.sampling_rate

    def retrieve(self, asr_embeddings: list | torch.Tensor, top_n=3):
        # 1. Convert PyTorch tensors to standard Python lists of lists for Chroma
        if isinstance(asr_embeddings, torch.Tensor):
            # Ensure it's 2D: [batch_size, embedding_dim]
            if asr_embeddings.ndim == 1:
                asr_embeddings = asr_embeddings.unsqueeze(0)
            embeddings_list = asr_embeddings.cpu().numpy().tolist()
        else:
            # Assume it's a list of lists or list of 1D tensors
            embeddings_list = [
                emb.cpu().numpy().tolist() if isinstance(emb, torch.Tensor) else emb
                for emb in asr_embeddings
            ]

        # 2. Query using 'query_embeddings' instead of 'query_texts'
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

    def inference(self, audios: list):
        is_single = not isinstance(audios, list)
        if is_single:
            audios = [audios]

        # 3. Retrieve using the ASR embeddings rather than the text prompt
        batched_examples = self.retrieve(audios)

        formatted_prompts = []
        processed_audios = []

        for i, audio in enumerate(audios):
            # Extract raw tensor from torchcodec object
            samples = audio.get_all_samples()
            audio = samples.data.squeeze(dim=0)

            if samples.sample_rate != self.target_sampling_rate:
                audio = F.resample(
                    audio,
                    orig_freq=samples.sample_rate,
                    new_freq=self.target_sampling_rate,
                )

            processed_audios.append(audio)

            examples_str = "\n\n".join(batched_examples[i])
            system_prompt = (
                f"{self.system_prompt}\n\n<examples>\n{examples_str}\n</examples>"
            )

            message = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": f"torchcodec_stream_{i}"},
                        {
                            "type": "text",
                            "text": "Transcribe this and fix the formatting",
                        },
                    ],
                },
            ]

            formatted_prompt = self.llm_processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)

        inputs = self.llm_processor(
            text=formatted_prompts,
            audios=processed_audios,
            return_tensors="pt",
            padding=True,
        ).to(self.llm_model.device)

        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=self.llm_processor.tokenizer.pad_token_id,
                eos_token_id=self.llm_processor.tokenizer.eos_token_id,
                temperature=0.2,
                do_sample=True,
            )

        generated_ids = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]

        res = self.llm_processor.batch_decode(generated_ids, skip_special_tokens=True)
        return res[0] if is_single else res
