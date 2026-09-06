import torchaudio
import torch

from data.normalizer import contains_equation, has_valid_equation
from transformers import AutoTokenizer
from data.filters import combined_filter

# 1. Add label=None to allow calling without arguments
def create_messages(label=None):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an expert transcription editor. Correct the following clip of audio for grammatical errors, mathematical formatting, and LaTeX terminology. Output ONLY the corrected text.",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio"},
            ],
        },
    ]

    if label:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": label}]}
        )

    return messages


def preprocess_speech2latex(dataset, processor, normalizer):
    target_sampling_rate = processor.feature_extractor.sampling_rate
    dataset = dataset.filter(combined_filter, num_proc=10)

    # Note: batched=False is used here since custom audio loading usually operates row-by-row
    def preprocess(sample):
        # 2. Extract and resample audio
        samples = sample["audio_path"].get_all_samples()
        audio = samples.data.squeeze(dim=0)

        if samples.sample_rate != target_sampling_rate:
            audio = torchaudio.functional.resample(
                audio, orig_freq=samples.sample_rate, new_freq=target_sampling_rate
            )

        label = sample["sentence"]

        if normalizer:
            label = normalizer(label)

        # 3. Create strings via chat template
        prompt_messages = create_messages(label=None)
        full_messages = create_messages(label=label)

        prompt_text = processor.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=False
        )
        full_text = processor.apply_chat_template(
            full_messages, add_generation_prompt=False, tokenize=False
        )

        # 4. Process Inputs (padding=False is mandatory here)
        # We pass audio.numpy() as Hugging Face processors typically prefer numpy for audio features
        full_inputs = processor(
            text=full_text, 
            audios=audio.numpy(), 
            sampling_rate=target_sampling_rate,
            return_tensors="pt", 
            padding=False 
        )
        
        # 5. Create labels array and mask out the prompt (using -100)
        # Tokenize just the prompt to find out how many tokens it takes
        prompt_inputs = processor(
            text=prompt_text,
            audios=audio.numpy(),
            return_tensors="pt",
            padding=False
        )
        prompt_length = prompt_inputs["input_ids"].shape[-1]
        
        # Copy input_ids to create the labels
        labels = full_inputs["input_ids"].clone()
        
        # Mask out the prompt tokens so the model only calculates loss on the generated label
        labels[0, :prompt_length] = -100

        # 6. Unroll dict and remove the fake batch dimension (squeeze 0)
        processed_sample = {
            "input_ids": full_inputs["input_ids"].squeeze(0),
            "attention_mask": full_inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "input": prompt_text,  # Kept string for inference loop to use
            "label": label,         # Kept string for WER/CER evaluation calculation
            "prompt_length": prompt_length
        }
        
        # Dynamically add audio specific features (e.g., QwenAudio outputs `audio_values`)
        # Squeeze them so the data collator can batch them properly later
        for key in full_inputs.keys():
            if key not in ["input_ids", "attention_mask"]:
                processed_sample[key] = full_inputs[key].squeeze(0)

        return processed_sample

    # Map across the entire dataset. remove_columns ensures we don't carry over unbatched old columns.
    dataset = dataset.map(preprocess, batched=False, remove_columns=dataset.column_names)
    return dataset