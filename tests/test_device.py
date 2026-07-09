"""Tests for device/compute-type selection logic."""

from transcript.device import default_compute_type, detect_device


def test_prefer_cpu_is_respected():
    assert detect_device("cpu") == "cpu"


def test_mps_is_never_returned():
    # CTranslate2 has no MPS backend; we must downgrade to CPU.
    assert detect_device("mps") == "cpu"


def test_compute_type_defaults():
    assert default_compute_type("cuda") == "float16"
    assert default_compute_type("cpu") == "int8"
