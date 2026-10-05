from rag_eval_lab.core.models import Chunk, EvaluationCase, EvaluationSample, GroundTruth, RetrievedChunk
from rag_eval_lab.evaluation.answer.base import available_inputs, close_evaluator, is_applicable


def _sample(generated_answer=None, reference_answer=None, retrieved_chunks=None) -> EvaluationSample:
    case = EvaluationCase(
        query_id="q1", question="q?", ground_truth=GroundTruth(reference_answer=reference_answer)
    )
    return EvaluationSample(
        evaluation_case=case, retrieved_chunks=retrieved_chunks or [], generated_answer=generated_answer
    )


def _retrieved_chunk() -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(id="c1", document_id="d1", text="text"), score=1.0, rank=1)


def test_available_inputs_reflects_sample_state():
    assert available_inputs(_sample()) == set()
    assert available_inputs(_sample(generated_answer="answer")) == {"generated_answer"}
    assert available_inputs(_sample(generated_answer="a", reference_answer="r")) == {
        "generated_answer",
        "reference_answer",
    }
    assert available_inputs(_sample(generated_answer="a", retrieved_chunks=[_retrieved_chunk()])) == {
        "generated_answer",
        "retrieved_context",
    }


def test_is_applicable_requires_every_declared_input():
    class _Evaluator:
        required_inputs = {"generated_answer", "reference_answer"}

    assert not is_applicable(_Evaluator(), _sample(generated_answer="a"))
    assert is_applicable(_Evaluator(), _sample(generated_answer="a", reference_answer="r"))


def test_close_evaluator_calls_close_only_if_present():
    calls = []

    class _WithClose:
        def close(self):
            calls.append("closed")

    class _WithoutClose:
        pass

    close_evaluator(_WithClose())
    close_evaluator(_WithoutClose())  # must not raise

    assert calls == ["closed"]
