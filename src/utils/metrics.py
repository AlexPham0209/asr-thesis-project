from utils.latex_metrics import LatexInContextMetrics
import torch
import numpy as np
import logging

logger = logging.getLogger("finetuning")

# Curried function for evaluating metrics
def create_metric(processor, normalizer=None):
    def compute_metrics(pred):
        pred_ids = pred.predictions

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        if isinstance(pred_ids, torch.Tensor) and pred_ids.ndim == 3:
            pred_ids = pred_ids.argmax(dim=-1)

        if not isinstance(pred_ids, torch.Tensor):
            pred_ids = torch.from_numpy(pred_ids)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(
            pred.label_ids, skip_special_tokens=True, group_tokens=False
        )

        metrics = LatexInContextMetrics(text_normalizer=normalizer)
        result = metrics.compute_all(pred_str, label_str)
        return result

    return compute_metrics


# Curried function for evaluating metrics
def create_llm_metric(tokenizer, normalizer=None):
    def compute_metrics(pred):
        pred_ids, label_ids = pred.predictions, pred.label_ids

        pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        metrics = LatexInContextMetrics(text_normalizer=normalizer)
        result = metrics.compute_all(pred_str, label_str)
        return result

    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    """
    Runs on the GPU after every evaluation step.
    Converts full (batch, seq_len, vocab_size) float32 logits into
    compact (batch, seq_len) int64 argmax token IDs before host transfer.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    
    # Take argmax on GPU to drop the heavy vocab dimension
    return logits.argmax(dim=-1)