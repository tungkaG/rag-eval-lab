"""NLI-based answer evaluator, wrapping a Hugging Face sequence classification model.

Compares the generated answer against either the reference answer
(`comparison="reference"`) or the concatenated final retrieved context
(`comparison="context"`) as a single whole-answer entailment judgment --
NOT claim-level verification. Long premises/hypotheses are truncated to the
model's input limit (recorded in metric metadata).

`transformers`/`torch` are optional (the `nli` extra) and imported lazily,
only when this evaluator is constructed, so retrieval-only experiments never
need them installed or any model downloaded.
"""

from __future__ import annotations

from typing import Literal

from rag_eval_lab.core.exceptions import ConfigurationError
from rag_eval_lab.core.models import EvaluationSample, MetricResult
from rag_eval_lab.evaluation.answer.base import AnswerEvaluationInput

_REQUIRED_LABELS = {"entailment", "contradiction", "neutral"}


class NliAnswerEvaluator:
    """Scores entailment/contradiction/neutral probabilities for one answer comparison."""

    def __init__(
        self,
        model_name: str,
        comparison: Literal["reference", "context"] = "reference",
        device: str = "cpu",
    ) -> None:
        if comparison not in ("reference", "context"):
            raise ConfigurationError(f"NLI 'comparison' must be 'reference' or 'context', got '{comparison}'")

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ConfigurationError(
                "transformers is required for the 'nli' answer evaluator; install it with "
                'pip install -e ".[nli]"'
            ) from exc

        self.model_name = model_name
        self.comparison = comparison
        self.device = device
        self.required_inputs: set[AnswerEvaluationInput] = (
            {"generated_answer", "reference_answer"}
            if comparison == "reference"
            else {"generated_answer", "retrieved_context"}
        )

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(device)
        self._model.eval()

        id2label = {index: str(label).lower() for index, label in self._model.config.id2label.items()}
        if set(id2label.values()) != _REQUIRED_LABELS:
            raise ConfigurationError(
                f"NLI model '{model_name}' must be a three-label model with labels "
                f"{sorted(_REQUIRED_LABELS)}, got {sorted(id2label.values())}"
            )
        self._label_index = {label: index for index, label in id2label.items()}

    def evaluate(self, sample: EvaluationSample) -> list[MetricResult]:
        import torch

        if self.comparison == "reference":
            premise = sample.evaluation_case.ground_truth.reference_answer
        else:
            premise = "\n\n".join(result.chunk.text for result in sample.retrieved_chunks)
        hypothesis = sample.generated_answer

        full_length = len(self._tokenizer(premise, hypothesis)["input_ids"])
        encoded = self._tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=self._tokenizer.model_max_length,
            return_tensors="pt",
        )
        truncated = full_length > encoded["input_ids"].shape[1]
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = self._model(**encoded).logits[0]
        probabilities = torch.softmax(logits, dim=-1).tolist()

        metadata = {"model_name": self.model_name, "comparison": self.comparison, "truncated": truncated}
        prefix = f"answer_nli_{self.comparison}"
        return [
            MetricResult(
                name=f"{prefix}_entailment",
                value=probabilities[self._label_index["entailment"]],
                k=None,
                metadata=metadata,
            ),
            MetricResult(
                name=f"{prefix}_contradiction",
                value=probabilities[self._label_index["contradiction"]],
                k=None,
                metadata=metadata,
            ),
            MetricResult(
                name=f"{prefix}_neutral",
                value=probabilities[self._label_index["neutral"]],
                k=None,
                metadata=metadata,
            ),
        ]
