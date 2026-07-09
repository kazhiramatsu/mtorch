from __future__ import annotations

import pytest

import mtorch


def test_cpu_device_keyword_creates_cpu_tensor() -> None:
    tensor = mtorch.zeros((2, 3), dtype=mtorch.float32, device=mtorch.device("cpu"))

    assert tensor.device == "cpu"


def test_metal_device_is_parsed_but_storage_is_not_implemented() -> None:
    assert str(mtorch.device("metal")) == "mps"

    with pytest.raises(NotImplementedError):
        mtorch.zeros((1,), device=mtorch.device("mps"))
