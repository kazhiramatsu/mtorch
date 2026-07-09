from __future__ import annotations

import pytest

from .harness import assert_view_alias_parity


pytestmark = [pytest.mark.compat, pytest.mark.mutation]


def test_view_aliasing_matches_reference(torch_reference, torch_candidate, view_case) -> None:
    assert_view_alias_parity(torch_reference, torch_candidate, view_case)
