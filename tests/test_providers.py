from surprisal.providers import LiteratureStatus, ProviderStatus


def test_both_available():
    status = ProviderStatus(claude_available=True, codex_available=True)
    assert status.any_available is True
    assert status.both_available is True


def test_claude_only():
    status = ProviderStatus(claude_available=True, codex_available=False)
    assert status.any_available is True
    assert status.both_available is False


def test_codex_only():
    status = ProviderStatus(claude_available=False, codex_available=True)
    assert status.any_available is True
    assert status.both_available is False


def test_neither_available():
    status = ProviderStatus(claude_available=False, codex_available=False)
    assert status.any_available is False
    assert status.both_available is False


def test_literature_alphaxiv():
    status = LiteratureStatus(provider="alphaxiv")
    assert status.has_semantic_search is True
    assert status.provider == "alphaxiv"


def test_literature_huggingface():
    status = LiteratureStatus(provider="huggingface")
    assert status.has_semantic_search is False


def test_literature_none():
    status = LiteratureStatus(provider="none")
    assert status.has_semantic_search is False
