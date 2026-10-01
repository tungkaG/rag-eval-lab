import pytest

from rag_eval_lab.core.exceptions import ValidationError
from rag_eval_lab.core.models import (
    Chunk,
    Document,
    EvaluationCase,
    EvaluationSample,
    ExperimentResult,
    GroundTruth,
    MetricResult,
    PageReference,
    Query,
    RetrievedChunk,
)


def test_document_round_trip():
    doc = Document(id="doc_1", text="hello world", metadata={"source": "txt"})
    assert doc.id == "doc_1"
    assert doc.text == "hello world"
    assert doc.metadata == {"source": "txt"}


def test_document_requires_non_empty_id():
    with pytest.raises(ValidationError):
        Document(id="", text="hello world")


def test_document_default_metadata_is_independent_dict():
    doc_a = Document(id="a", text="x")
    doc_b = Document(id="b", text="y")
    doc_a.metadata["k"] = "v"
    assert doc_b.metadata == {}


def test_chunk_round_trip():
    chunk = Chunk(
        id="doc_1_chunk_0",
        document_id="doc_1",
        text="hello",
        page_start=1,
        page_end=2,
        char_start=0,
        char_end=5,
    )
    assert chunk.document_id == "doc_1"
    assert chunk.page_start == 1
    assert chunk.page_end == 2


def test_chunk_requires_non_empty_ids():
    with pytest.raises(ValidationError):
        Chunk(id="", document_id="doc_1", text="hello")
    with pytest.raises(ValidationError):
        Chunk(id="chunk_0", document_id="", text="hello")


def test_chunk_rejects_inverted_page_range():
    with pytest.raises(ValidationError):
        Chunk(id="c", document_id="d", text="t", page_start=5, page_end=1)


def test_chunk_rejects_inverted_char_range():
    with pytest.raises(ValidationError):
        Chunk(id="c", document_id="d", text="t", char_start=100, char_end=1)


def test_chunk_allows_missing_page_and_char_offsets():
    chunk = Chunk(id="c", document_id="d", text="t")
    assert chunk.page_start is None
    assert chunk.char_end is None


def test_query_requires_non_empty_id():
    with pytest.raises(ValidationError):
        Query(id="", text="what is x?")


def test_page_reference_round_trip():
    ref = PageReference(document_id="doc_1", page_start=23, page_end=23)
    assert ref.page_start == 23


def test_page_reference_rejects_inverted_range():
    with pytest.raises(ValidationError):
        PageReference(document_id="doc_1", page_start=10, page_end=5)


def test_ground_truth_defaults_are_empty_and_independent():
    gt_a = GroundTruth()
    gt_b = GroundTruth()
    gt_a.document_relevance["doc_1"] = 1
    assert gt_b.document_relevance == {}
    assert gt_a.reference_answer is None


def test_ground_truth_document_relevance_binary():
    gt = GroundTruth(document_relevance={"doc_1": 1, "doc_2": 0})
    assert gt.document_relevance["doc_1"] == 1
    assert gt.document_relevance["doc_2"] == 0
    relevant = [doc_id for doc_id, grade in gt.document_relevance.items() if grade > 0]
    assert relevant == ["doc_1"]


def test_ground_truth_document_relevance_graded():
    gt = GroundTruth(document_relevance={"doc_1": 3, "doc_2": 1, "doc_3": 0})
    relevant = {doc_id for doc_id, grade in gt.document_relevance.items() if grade > 0}
    assert relevant == {"doc_1", "doc_2"}
    assert gt.document_relevance["doc_1"] == 3


def test_ground_truth_capabilities_empty():
    assert GroundTruth().capabilities == set()


def test_ground_truth_capabilities_document_relevance_only():
    gt = GroundTruth(document_relevance={"doc_1": 1})
    assert gt.capabilities == {"document_relevance"}


def test_ground_truth_capabilities_page_references_only():
    gt = GroundTruth(page_references=[PageReference(document_id="doc_1", page_start=1, page_end=1)])
    assert gt.capabilities == {"page_references"}


def test_ground_truth_capabilities_blank_reference_answer_is_not_counted():
    assert GroundTruth(reference_answer="").capabilities == set()
    assert GroundTruth(reference_answer="   ").capabilities == set()


def test_ground_truth_capabilities_non_blank_reference_answer_is_counted():
    assert GroundTruth(reference_answer="85 degrees Celsius").capabilities == {"reference_answer"}


def test_ground_truth_capabilities_evidence_and_keywords():
    gt = GroundTruth(evidence_texts=["the device operates up to 85C"], keywords=["temperature"])
    assert gt.capabilities == {"evidence_texts", "keywords"}


def test_ground_truth_capabilities_combination_is_recomputed_live():
    gt = GroundTruth(document_relevance={"doc_1": 1})
    assert gt.capabilities == {"document_relevance"}
    gt.reference_answer = "85 degrees Celsius"
    assert gt.capabilities == {"document_relevance", "reference_answer"}


def test_evaluation_case_requires_question_and_query_id():
    gt = GroundTruth(document_relevance={"doc_1": 1})
    with pytest.raises(ValidationError):
        EvaluationCase(query_id="", question="q?", ground_truth=gt)
    with pytest.raises(ValidationError):
        EvaluationCase(query_id="q1", question="", ground_truth=gt)


def test_retrieved_chunk_round_trip():
    chunk = Chunk(id="c", document_id="d", text="t")
    retrieved = RetrievedChunk(chunk=chunk, score=0.9, rank=0)
    assert retrieved.rank == 0
    assert retrieved.score == 0.9


def test_retrieved_chunk_rejects_negative_rank():
    chunk = Chunk(id="c", document_id="d", text="t")
    with pytest.raises(ValidationError):
        RetrievedChunk(chunk=chunk, score=0.9, rank=-1)


def test_evaluation_sample_defaults():
    case = EvaluationCase(query_id="q1", question="q?", ground_truth=GroundTruth())
    sample = EvaluationSample(evaluation_case=case)
    assert sample.retrieved_chunks == []
    assert sample.generated_answer is None
    assert sample.timings == {}


def test_metric_result_requires_non_empty_name():
    with pytest.raises(ValidationError):
        MetricResult(name="", value=1.0)


def test_metric_result_round_trip():
    result = MetricResult(name="recall@10", value=0.8, k=10)
    assert result.k == 10


def test_experiment_result_requires_non_empty_id():
    with pytest.raises(ValidationError):
        ExperimentResult(experiment_id="", config={})


def test_experiment_result_round_trip():
    metric = MetricResult(name="recall@10", value=0.8, k=10)
    result = ExperimentResult(
        experiment_id="exp_1",
        config={"chunking": {"strategy": "fixed_tokens"}},
        metrics=[metric],
    )
    assert result.metrics[0].value == 0.8
