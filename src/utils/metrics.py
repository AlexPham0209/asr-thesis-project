import evaluate
import numpy as np

# Metrics
wer = evaluate.load("wer")
cer = evaluate.load("cer")


# Curried function for evaluating metrics
def create_metric(processor, normalizer=None):
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(
            pred.label_ids, skip_special_tokens=True, group_tokens=False
        )

        wer_ortho = 100 * wer.compute(predictions=pred_str, references=label_str)
        cer_ortho = 100 * cer.compute(predictions=pred_str, references=label_str)

        results = {}

        results["wer_ortho"] = wer_ortho
        results["cer_ortho"] = cer_ortho

        if normalizer:
            pred_str_norm = list(map(lambda str: normalizer(str), pred_str))
            label_str_norm = list(map(lambda str: normalizer(str), label_str))

            wer_score = 100 * wer.compute(
                predictions=pred_str_norm, references=label_str_norm
            )
            cer_score = 100 * cer.compute(
                predictions=pred_str_norm, references=label_str_norm
            )

            results["wer"] = wer_score
            results["cer"] = cer_score

        return results

    return compute_metrics
