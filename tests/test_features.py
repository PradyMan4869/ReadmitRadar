"""Feature schema and synthetic-data invariants."""
import pytest

from data.synthetic import generate_hospital
from ml.features import (
    BINARY_FEATURES, FEATURE_COLUMNS, RISK_BANDS, TARGET, validate_row,
)


def test_schema_columns_present(hospital_a, hospital_b):
    for df in (hospital_a, hospital_b):
        for col in FEATURE_COLUMNS + [TARGET]:
            assert col in df.columns, col


def test_binary_features_are_binary(hospital_a):
    for col in BINARY_FEATURES:
        assert set(hospital_a[col].unique()) <= {0, 1}, col


def test_case_mix_differs_between_hospitals(hospital_a, hospital_b):
    # Hospital A is the older, cardiac-heavy site by construction
    assert hospital_a["age"].mean() > hospital_b["age"].mean() + 5
    assert hospital_a["has_chf"].mean() > hospital_b["has_chf"].mean()
    assert hospital_b["has_diabetes"].mean() > hospital_a["has_diabetes"].mean()


def test_label_rate_plausible(hospital_a, hospital_b):
    # 30-day readmission rates in literature: roughly 10–30%
    for df in (hospital_a, hospital_b):
        assert 0.05 < df[TARGET].mean() < 0.45


def test_generation_is_deterministic():
    a1 = generate_hospital("A", 50, seed=3)
    a2 = generate_hospital("A", 50, seed=3)
    assert a1.equals(a2)


def test_validate_row_catches_problems(hospital_a):
    row = hospital_a.iloc[0].to_dict()
    assert validate_row(row) == []
    bad = dict(row)
    bad["age"] = 300
    del bad["bun_last"]
    problems = validate_row(bad)
    assert any("age" in p for p in problems)
    assert any("bun_last" in p for p in problems)


def test_risk_bands():
    assert RISK_BANDS.label(0.1) == "LOW"
    assert RISK_BANDS.label(0.25) == "MODERATE"
    assert RISK_BANDS.label(0.45) == "ELEVATED"
    assert RISK_BANDS.label(0.8) == "HIGH"
