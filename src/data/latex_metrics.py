import logging
from typing import List
from tqdm import tqdm
import pandas as pd

import evaluate

from data.normalize_equation import NormalizeFormula

logger = logging.getLogger("finetuning")


class LatexInContextMetrics:
    def __init__(self, text_normalizer=None, equation_normalizer=None):
        self.wer = evaluate.load("wer")
        self.cer = evaluate.load("cer")

        self.bleu = evaluate.load("bleu")
        self.meteor = evaluate.load("meteor")
        self.sacrebleu = evaluate.load("sacrebleu")
        self.rouge1 = evaluate.load("rouge")
        self.chrf = evaluate.load("chrf")
        self.text_normalizer = text_normalizer
        self.equation_normalizer = equation_normalizer

    def compute(self, predictions, references, normalizer=None):
        predictions_lower = [x.lower() for x in predictions]
        references_lower = [x.lower() for x in references]

        # Normalized text
        predictions_normalized = None
        references_normalized = None
        if normalizer:
            predictions_normalized = [normalizer(x) for x in predictions]
            references_normalized = [normalizer(x) for x in references]

        result = {}

        # Word Error Rate
        try:
            result["wer"] = self.wer.compute(
                predictions=predictions, references=references
            )
            result["wer_lower"] = self.wer.compute(
                predictions=predictions_lower, references=references_lower
            )
            if normalizer:
                result["wer_normalized"] = self.wer.compute(
                    predictions=predictions_normalized, references=references_normalized
                )
        except Exception as e:
            logger.error(f"Error computing wer: {e}")
            result["wer"] = -1
            result["wer_lower"] = -1

        # Character Error Rate
        result["cer"] = self.cer.compute(predictions=predictions, references=references)
        result["cer_lower"] = self.cer.compute(
            predictions=predictions_lower, references=references_lower
        )

        if normalizer:
            result["cer_normalized"] = self.cer.compute(
                predictions=predictions_normalized, references=references_normalized
            )

        # Other metrics
        result["bleu"] = self.bleu.compute(
            predictions=predictions, references=references
        )["bleu"]
        result["sacrebleu"] = (
            self.sacrebleu.compute(predictions=predictions, references=references)[
                "score"
            ]
            / 100
        )
        result["meteor"] = self.meteor.compute(
            predictions=predictions, references=references
        )["meteor"].item()

        result["rouge1"] = self.rouge1.compute(
            predictions=predictions, references=references
        )["rouge1"].item()

        result["chrf"] = (
            self.chrf.compute(predictions=predictions, references=references)["score"]
            / 100
        )

        result["chrfpp"] = (
            self.chrf.compute(
                predictions=predictions, references=references, word_order=2
            )["score"]
            / 100
        )

        return result

    def compute_formulas_only(self, predictions, references):
        """
        Extracts all formulas from predictions and references,
        concatenate all formulas. And computes metrics for formulas-only string.
        """
        prediction_formulas_only, formulas_content_list = (
            self.extract_in_context_formulas_bulk(predictions)
        )

        references_formulas_only, _ = self.extract_in_context_formulas_bulk(references)
        metrics = self.compute(
            prediction_formulas_only,
            references_formulas_only,
        )

        if self.equation_normalizer:
            normalized_formulas = self.equation_normalizer(formulas_content_list)
            invalid_count = sum(1 for x in normalized_formulas if x == "")

            metrics["invalid_latex"] = invalid_count / len(formulas_content_list)

        return metrics

    def extract_in_context_formulas_bulk(self, text_lines: List[str]):
        result = []
        formulas_content_list = []
        for text_line in text_lines:
            formulas_content = text_line.split("$")
            if len(formulas_content) < 2:
                formulas_content = [""]
            else:
                formulas_content = formulas_content[1::2]

            formulas_content_list.extend(formulas_content)
            result.append("$" + "$ $".join(formulas_content) + "$")

        return result, formulas_content_list

    def extract_in_context_text_bulk(self, text_lines: List[str]):
        result = []
        for text_line in text_lines:
            formulas_content = text_line.split("$")
            if len(formulas_content) < 2:
                formulas_content = [text_line]
            else:
                formulas_content = formulas_content[0::2]

            result.append(" ".join(formulas_content))

        return result

    def compute_text_only(self, predictions, references):
        """
        Extracts all formulas from predictions and references,
        concatenate all formulas. And computes metrics for formulas-only string.
        """
        prediction_text_only = self.extract_in_context_text_bulk(predictions)
        references_text_only = self.extract_in_context_text_bulk(references)
        metrics = self.compute(
            prediction_text_only, references_text_only, normalizer=self.text_normalizer
        )

        return metrics

    def compute_all(
        self,
        predictions,
        references,
        compute_text_only=True,
        compute_formulas_only=True,
        normalize=False,
    ):
        if isinstance(predictions, pd.Series):
            predictions = predictions.values.tolist()
        if isinstance(references, pd.Series):
            references = references.values.tolist()

        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]

        metrics = self.compute(predictions, references, self.text_normalizer)

        if compute_formulas_only:
            metrics_formulas_only = self.compute_formulas_only(predictions, references)
            for k, v in metrics_formulas_only.items():
                metrics["formula_" + k] = v

        if compute_text_only:
            metrics_text_only = self.compute_text_only(predictions, references)
            for k, v in metrics_text_only.items():
                metrics["text_" + k] = v

        return metrics
