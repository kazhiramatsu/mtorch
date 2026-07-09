from __future__ import annotations

import pytest

from .harness import api_kind, maybe_signature, resolve_path


pytestmark = [pytest.mark.compat, pytest.mark.api_surface]


def test_public_api_entry_exists_and_matches_kind(torch_reference, torch_candidate, api_entry) -> None:
    expected = resolve_path(torch_reference, api_entry["path"])
    actual = resolve_path(torch_candidate, api_entry["path"])

    expected_kind = api_entry.get("kind") or api_kind(expected)
    actual_kind = api_kind(actual)
    assert actual_kind == expected_kind

    expected_signature = api_entry.get("signature")
    if expected_signature:
        assert maybe_signature(actual) == expected_signature
