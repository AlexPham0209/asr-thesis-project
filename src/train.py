import evaluate
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModelForCTC
import numpy as np
from hydra.utils import instantiate
from datasets import load_dataset

from data.data_collator import DataCollatorCTCWithPadding, DataCollatorSpeechSeq2SeqWithPadding

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


def create_seq2seq_trainer(
    cfg, model, processor, train, valid, compute_metrics, data_collator
):
    training_args = Seq2SeqTrainingArguments(**cfg.training)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train,
        eval_dataset=valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    return trainer


def create_ctc_trainer(
    cfg, model, processor, train, valid, compute_metrics, data_collator
):
    print(cfg.training)
    training_args = TrainingArguments(**cfg.training)

    # 6. Initialize the Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        processing_class=processor,
        compute_metrics=compute_metrics,
    )

    return trainer

def create_custom_trainer(cfg, model, processor, train, test, compute_metrics, data_collator):
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
    processor = hydra.utils.instantiate(cfg.processor)
    model = hydra.utils.instantiate(cfg.model).to(device)

    # Creating Dataset and Dataloader
    if not cfg.get("dataset"):
        raise ValueError("Missing 'data' configutation block in your YAML")

    # Loading in dataset
    datasets = hydra.utils.instantiate(cfg.dataset)
    train, valid, test = datasets.train, datasets.validation, datasets.test

    # Instantiating preprocessing function an then preprocessing the raw dataset
    # Each sample should be in the following format: {input_features/input_values, labels, input_lengths}
    preprocess_fn = hydra.utils.instantiate(cfg.preprocess, processor=processor)
    train = preprocess_fn(train)
    valid = preprocess_fn(valid)
    test = preprocess_fn(test)

    # Creating metrics
    compute_metrics = create_metric(processor=processor)

    # Creating trainer
    architecture = cfg.architecture
    trainer = (
        create_ctc_trainer(
            cfg=cfg,
            model=model,
            processor=processor,
            train=train,
            valid=valid,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorCTCWithPadding(processor=processor)
        )
        if architecture == "ctc"
        else create_seq2seq_trainer(
            cfg=cfg,
            model=model,
            processor=processor,
            train=train,
            valid=valid,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        )
    )

    print(trainer)

    # trainer.train()


if __name__ == "__main__":
    main()
