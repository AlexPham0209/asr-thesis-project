import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperProcessor
from datasets import load_dataset
import transformers
import datasets

from data.filters import combined_filter
from utils.latex_metrics import LatexInContextMetrics

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    pretrained_model_name_or_path="openai/whisper-small", device_map="auto"
)
processor: WhisperProcessor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path="openai/whisper-small"
)
target_sampling_rate = processor.feature_extractor.sampling_rate

# 3. Corrected and vectorized batched mapping
def evaluate_batch(batch):
    audios = []

    # Process the batched torchcodec objects
    for audio in batch["audio_path"]:
        samples = audio.get_all_samples()

        # Remove the channel dimension: [1, frames] -> [frames]
        audio_tensor = samples.data.squeeze(dim=0)

        # Whisper requires exactly 16000Hz. If the native audio is different, resample it.
        if samples.sample_rate != target_sampling_rate:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor,
                orig_freq=samples.sample_rate,
                new_freq=target_sampling_rate,
            )

        # Convert to numpy for the HF processor
        audios.append(audio_tensor.numpy())

    references = batch["whisper_text"]

    # Run HF Processor
    inputs = processor(
        audio=audios, sampling_rate=target_sampling_rate, return_tensors="pt"
    ).to(device)

    # Generate predictions
    with torch.no_grad():
        generated_ids = model.generate(inputs["input_features"])

    # Decode the model outputs to text
    transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

    batch["predictions"] = transcriptions
    batch["references"] = references
    return batch


dataset = datasets.load_dataset(
    "marsianin500/Speech2Latex", name="default", split="sentences_test"
)
dataset = dataset.filter(combined_filter, num_proc=10)
dataset = dataset.select(range(50)).map(evaluate_batch, batched=True, batch_size=16)

metrics = LatexInContextMetrics()

print(
    metrics.compute_all(
        predictions=dataset["predictions"], references=dataset["references"]
    )
)
