import rag_eval_lab
import rag_eval_lab.core
from rag_eval_lab.core.exceptions import RagEvalLabError


def test_package_imports_successfully():
    assert rag_eval_lab.__version__


def test_exception_hierarchy():
    assert issubclass(rag_eval_lab.core.ValidationError, RagEvalLabError)
    assert issubclass(rag_eval_lab.core.RegistryError, RagEvalLabError)
    assert issubclass(rag_eval_lab.core.UnknownStrategyError, rag_eval_lab.core.RegistryError)
    assert issubclass(rag_eval_lab.core.ConfigurationError, RagEvalLabError)
