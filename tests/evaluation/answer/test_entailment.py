"""Light unit test for `NliAnswerEvaluator` -- no model download (invalid config fails first)."""

import pytest

from rag_eval_lab.core.exceptions import ConfigurationError
from rag_eval_lab.evaluation.answer.entailment import NliAnswerEvaluator


def test_invalid_comparison_raises_before_loading_any_model():
    with pytest.raises(ConfigurationError):
        NliAnswerEvaluator(model_name="irrelevant", comparison="not_a_real_mode")
