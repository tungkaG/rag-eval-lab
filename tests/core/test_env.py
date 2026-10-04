"""Light smoke test for `core.env.load_env_file`."""

from __future__ import annotations

from rag_eval_lab.core.env import load_env_file


def test_load_env_file_is_safe_to_call_repeatedly() -> None:
    load_env_file()
    load_env_file()  # idempotent; must not raise even without python-dotenv or a .env file
