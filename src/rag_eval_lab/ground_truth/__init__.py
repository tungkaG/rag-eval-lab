"""Ground truth helpers: native record conversion and basic relation checks.

`GroundTruth`, `PageReference`, and `GroundTruthCapability` live in
`rag_eval_lab.core.models`; this package only adds small, reusable
conversion and validation helpers on top of them.
"""

from rag_eval_lab.ground_truth.adapters import evaluation_case_from_record, ground_truth_from_record
from rag_eval_lab.ground_truth.validator import check_document_references_exist, normalize_ground_truth

__all__ = [
    "check_document_references_exist",
    "evaluation_case_from_record",
    "ground_truth_from_record",
    "normalize_ground_truth",
]
