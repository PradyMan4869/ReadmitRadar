"""FHIR layer: build → parse round-trip and validation failure modes."""
import pytest

from fhir.builder import build_bundle
from fhir.parser import BundleParseError, parse_bundle
from ml.features import FEATURE_COLUMNS


def test_bundle_shape(hospital_a):
    record = hospital_a.iloc[0].to_dict()
    bundle = build_bundle(record)
    assert bundle["resourceType"] == "Bundle"
    types = {e["resource"]["resourceType"] for e in bundle["entry"]}
    assert {"Patient", "Encounter", "Observation"} <= types
    # Conditions only appear when the comorbidity flag is set
    n_conditions = sum(
        1 for e in bundle["entry"]
        if e["resource"]["resourceType"] == "Condition"
    )
    flags = ["has_chf", "has_copd", "has_diabetes", "has_renal_disease"]
    assert n_conditions == sum(int(record[f]) for f in flags)


def test_round_trip_preserves_features(hospital_a):
    for _, rec in hospital_a.head(25).iterrows():
        record = rec.to_dict()
        row = parse_bundle(build_bundle(record))
        for col in FEATURE_COLUMNS:
            assert float(row[col]) == pytest.approx(float(record[col])), col


def test_parse_rejects_non_bundle():
    with pytest.raises(BundleParseError, match="resourceType"):
        parse_bundle({"resourceType": "Patient"})


def test_parse_reports_missing_features(hospital_a):
    bundle = build_bundle(hospital_a.iloc[0].to_dict())
    # Drop all observations → labs and utilization features missing
    bundle["entry"] = [
        e for e in bundle["entry"]
        if e["resource"]["resourceType"] != "Observation"
    ]
    with pytest.raises(BundleParseError, match="missing feature"):
        parse_bundle(bundle)


def test_parser_ignores_unknown_resources(hospital_a):
    bundle = build_bundle(hospital_a.iloc[0].to_dict())
    bundle["entry"].append({"resource": {"resourceType": "MedicationRequest",
                                         "id": "x"}})
    parse_bundle(bundle)  # must not raise
