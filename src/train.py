import evaluate
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from transformers import AutoTokenizer, Seq2SeqTrainer, Trainer, TrainingArguments
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModelForCTC
import numpy as np
from hydra.utils import instantiate
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# Metrics
wer = evaluate.load("wer")
cer = evaluate.load("cer")


# Curried function for evaluating metrics
def create_metric(processor):
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer_score = wer.compute(predictions=pred_str, references=label_str)
        cer_score = cer.compute(predictions=pred_str, references=label_str)

        return {"wer": wer_score, "cer": cer_score}
    
    return compute_metrics


def inference():
    pass


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("------- Running Experiment Configuration -------")
    print(OmegaConf.to_yaml(cfg))

    if not cfg.get("model"):
        raise ValueError("Missing 'model' configutation block in your YAML")
    
    # Instantiating model and processor (Can either be a pretrained model or customly trained model)
    print("------- Model Configurations -------")
    print(f"{cfg.model}\n")
    
    if not cfg.get("processor"):
        raise ValueError("Missing 'model' configuration block in your YAML")

    print(f"{cfg.processor}\n")
    
    print("------- Instantiating Model from Configuration -------")
    processor: AutoProcessor = hydra.utils.instantiate(cfg.processor)
    model: nn.Module = hydra.utils.instantiate(cfg.model)
    
    # Creating Dataset and Dataloader
    if not cfg.get("dataset"):
        raise ValueError("Missing 'data' configutation block in your YAML")
    
    # Loading dataset and specific preprocessing function
    raw_dataset = hydra.utils.instantiate(cfg.dataset)

    # Instantiating preprocessing function and preprocess dataset
    preprocess_fn = hydra.utils.instantiate(cfg.preprocess, processor=processor)
    processed_dataset = preprocess_fn(raw_dataset)

    # Creating metrics
    compute_metrics = create_metric(processor=processor)
        


    
if __name__ == "__main__":
    main()
