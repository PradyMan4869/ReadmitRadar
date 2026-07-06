"""Synthea loader: 30-day label windowing and feature derivation on a
hand-built miniature run directory."""
import pandas as pd
import pytest

from data.synthea_loader import load_run
from ml.features import TARGET


def _write_run(tmp_path, encounters):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir(parents=True)

    pd.DataFrame([
        {"Id": "p1", "BIRTHDATE": "1950-01-01", "GENDER": "M"},
        {"Id": "p2", "BIRTHDATE": "1960-06-15", "GENDER": "F"},
    ]).to_csv(csv_dir / "patients.csv", index=False)

    pd.DataFrame(encounters).to_csv(csv_dir / "encounters.csv", index=False)

    pd.DataFrame([
        {"PATIENT": "p1", "START": "2020-01-01T00:00Z", "STOP": None,
         "DESCRIPTION": "Chronic congestive heart failure (disorder)"},
        {"PATIENT": "p2", "START": "2020-01-01T00:00Z", "STOP": None,
         "DESCRIPTION": "Diabetes mellitus type 2 (disorder)"},
    ]).to_csv(csv_dir / "conditions.csv", index=False)

    pd.DataFrame([
        {"PATIENT": "p1", "DATE": "2021-03-01T00:00Z", "CODE": "3094-0",
         "DESCRIPTION": "Urea nitrogen [Mass/volume] in Blood", "VALUE": "28"},
        {"PATIENT": "p1", "DATE": "2021-03-01T00:00Z", "CODE": "2160-0",
         "DESCRIPTION": "Creatinine [Mass/volume] in Blood", "VALUE": "1.4"},
    ]).to_csv(csv_dir / "observations.csv", index=False)

    pd.DataFrame([
        {"PATIENT": "p1", "START": "2021-01-01T00:00Z", "STOP": None,
         "DESCRIPTION": "Furosemide 40 MG Oral Tablet"},
    ]).to_csv(csv_dir / "medications.csv", index=False)
    return tmp_path


def _enc(pid, cls, start, stop):
    return {"Id": f"{pid}-{start}", "PATIENT": pid, "ENCOUNTERCLASS": cls,
            "START": start, "STOP": stop}


def test_readmission_label_windowing(tmp_path):
    run = _write_run(tmp_path, [
        # p1: admission, readmitted 10 days later → label 1 on the first
        _enc("p1", "inpatient", "2021-03-01T00:00Z", "2021-03-05T00:00Z"),
        _enc("p1", "inpatient", "2021-03-15T00:00Z", "2021-03-20T00:00Z"),
        # p2: admission with next one 90 days later → label 0
        _enc("p2", "inpatient", "2021-05-01T00:00Z", "2021-05-04T00:00Z"),
        _enc("p2", "inpatient", "2021-08-10T00:00Z", "2021-08-12T00:00Z"),
        # trailing wellness visits extend both records past the label window
        _enc("p1", "wellness", "2021-06-01T00:00Z", "2021-06-01T01:00Z"),
        _enc("p2", "wellness", "2021-10-01T00:00Z", "2021-10-01T01:00Z"),
    ])
    df, stats = load_run(run, "T")

    by_pid = {row["patient_id"].split("-")[1]: row
              for _, row in df.iterrows() if True}
    p1_first = df[df["patient_id"].str.contains("2021-03-01")].iloc[0]
    p2_first = df[df["patient_id"].str.contains("2021-05-01")].iloc[0]
    assert p1_first[TARGET] == 1
    assert p2_first[TARGET] == 0


def test_features_derived(tmp_path):
    run = _write_run(tmp_path, [
        _enc("p1", "emergency", "2021-01-10T00:00Z", "2021-01-10T06:00Z"),
        _enc("p1", "inpatient", "2021-03-01T00:00Z", "2021-03-05T00:00Z"),
        # follow-up visit 7 days post-discharge
        _enc("p1", "ambulatory", "2021-03-12T00:00Z", "2021-03-12T01:00Z"),
        _enc("p1", "wellness", "2021-08-01T00:00Z", "2021-08-01T01:00Z"),
        _enc("p2", "inpatient", "2021-05-01T00:00Z", "2021-05-04T00:00Z"),
        _enc("p2", "wellness", "2021-10-01T00:00Z", "2021-10-01T01:00Z"),
    ])
    df, stats = load_run(run, "T")
    p1 = df[df["gender_male"] == 1].iloc[0]

    assert p1["has_chf"] == 1
    assert p1["ed_visits_6mo"] == 1
    assert p1["followup_scheduled"] == 1
    assert p1["length_of_stay"] == pytest.approx(4.0)
    assert p1["bun_last"] == pytest.approx(28.0)
    assert p1["n_medications"] == 1

    p2 = df[df["gender_male"] == 0].iloc[0]
    assert p2["has_diabetes"] == 1
    assert p2["has_chf"] == 0
    # p2 has no labs → imputed with cohort median (p1's values)
    assert p2["creatinine_last"] == pytest.approx(1.4)
    assert stats["lab_imputed_pct"]["creatinine_last"] == 50.0
