from data.latex_metrics import LatexInContextMetrics
import torch


# Curried function for evaluating metrics
def create_metric(processor, normalizer=None):
    def compute_metrics(pred):
        pred_ids = pred.predictions

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        if isinstance(pred_ids, torch.Tensor) and pred_ids.ndim == 3:
            pred_ids = pred_ids.argmax(dim=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(
            pred.label_ids, skip_special_tokens=True, group_tokens=False
        )

        metrics = LatexInContextMetrics(text_normalizer=normalizer)
        result = metrics.compute_all(pred_str, label_str)
        return result

    return compute_metrics
