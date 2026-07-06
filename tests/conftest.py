import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from data.synthetic import generate_hospital, generate_test_set


@pytest.fixture(scope="session")
def hospital_a():
    return generate_hospital("A", 1200, seed=7)


@pytest.fixture(scope="session")
def hospital_b():
    return generate_hospital("B", 1200, seed=8)


@pytest.fixture(scope="session")
def test_set():
    return generate_test_set(600, seed=9)
