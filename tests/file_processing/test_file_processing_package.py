from rag_eval_lab.file_processing import FileProcessor, file_processor_registry


def test_package_imports_successfully():
    assert FileProcessor is not None
    assert file_processor_registry is not None


def test_registry_is_initially_empty():
    assert file_processor_registry.keys() == []
