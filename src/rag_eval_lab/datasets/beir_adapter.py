"""Adapter for locally extracted BEIR format datasets.

All BEIR specific file layout knowledge (corpus.jsonl, queries.jsonl,
qrels/{split}.tsv) is contained in this module. No other component knows
that BEIR exists.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rag_eval_lab.core.models import Document, EvaluationCase, GroundTruth
from rag_eval_lab.ground_truth.validator import check_document_references_exist, normalize_ground_truth
from rag_eval_lab.ingestion.jsonl_loader import JsonlLoader


class BeirDatasetAdapter:
    """Reads a locally extracted BEIR dataset directory.

    Expects `corpus.jsonl`, `queries.jsonl`, and `qrels/{split}.tsv` under
    `dataset_dir`. Does not download datasets or depend on the BEIR package.
    """

    def __init__(self, dataset_dir: str | Path, split: str = "test") -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self._documents: list[Document] | None = None
        self._queries: dict[str, str] | None = None
        self._evaluation_cases: list[EvaluationCase] | None = None

    def documents(self) -> list[Document]:
        if self._documents is None:
            loader = JsonlLoader(fields={"id": "_id", "text": "text", "title": "title"})
            self._documents = loader.load(self.dataset_dir / "corpus.jsonl")
        return self._documents

    def evaluation_cases(self) -> list[EvaluationCase]:
        if self._evaluation_cases is None:
            documents = self.documents()
            queries = self._load_queries()
            cases = []
            for query_id, document_relevance in self._load_qrels().items():
                ground_truth = GroundTruth(document_relevance=document_relevance)
                normalize_ground_truth(ground_truth)
                check_document_references_exist(ground_truth, documents)
                cases.append(
                    EvaluationCase(
                        query_id=query_id,
                        question=queries[query_id],
                        ground_truth=ground_truth,
                    )
                )
            self._evaluation_cases = cases
        return self._evaluation_cases

    def _load_queries(self) -> dict[str, str]:
        if self._queries is None:
            queries: dict[str, str] = {}
            with (self.dataset_dir / "queries.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    queries[str(record["_id"])] = record["text"]
            self._queries = queries
        return self._queries

    def _load_qrels(self) -> dict[str, dict[str, int]]:
        relevance: dict[str, dict[str, int]] = {}
        qrels_path = self.dataset_dir / "qrels" / f"{self.split}.tsv"
        with qrels_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                query_id = row["query-id"]
                corpus_id = row["corpus-id"]
                relevance.setdefault(query_id, {})[corpus_id] = int(row["score"])
        return relevance
