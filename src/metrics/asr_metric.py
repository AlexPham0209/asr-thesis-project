import evaluate
import numpy as np

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
