"""
Pytest configuration and shared fixtures.

Tests never call real LLMs: `.env` may hold real provider keys (load_dotenv runs
on import), so LLM features are disabled and the deterministic hashing embedder
is used. Set MEMORABLE_TEST_REAL_EMBEDDER=1 to exercise sentence-transformers.
"""

import os

os.environ["MEMORABLE_LLM__DEFAULT_MODEL"] = "none"
os.environ["MEMORABLE_EXTRACTION__MODE"] = "hybrid"
os.environ["MEMORABLE_EXTRACTION__ADD_ONLY"] = "false"
os.environ["MEMORABLE_MEMORY__WRITE_MODE"] = "sync"
os.environ.pop("MEMORABLE_DATABASE__CONNECTION_STRING", None)
if not os.environ.get("MEMORABLE_TEST_REAL_EMBEDDER"):
    os.environ["MEMORABLE_LLM__EMBEDDING_MODEL"] = "hash"

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def test_environment():
    """Set up test environment variables."""
    os.environ["MEMORABLE_DATABASE__CONNECTION_STRING"] = "sqlite:///test.db"
    os.environ["MEMORABLE_MEMORY__MODE"] = "auto"
    yield
    if os.path.exists("test.db"):
        os.remove("test.db")


@pytest.fixture
def engine_factory(tmp_path):
    """Build MemoryEngines on a temp SQLite file; disables them afterwards."""
    from memorable_ai import MemoryEngine

    engines = []

    def make(**kwargs):
        kwargs.setdefault("database", f"sqlite:///{tmp_path / 'mem.db'}")
        kwargs.setdefault("mode", "auto")
        kwargs.setdefault("write_mode", "sync")
        eng = MemoryEngine(**kwargs)
        engines.append(eng)
        return eng

    yield make
    for eng in engines:
        try:
            eng.close()
        except Exception:
            pass


@pytest.fixture(autouse=True)
def _reset_interceptors():
    """Some legacy tests enable engines without disabling them; unpatch SDKs between tests."""
    yield
    from memorable_ai.core import interceptor as icp

    for interceptor in list(icp._STACK):
        interceptor.disable()
