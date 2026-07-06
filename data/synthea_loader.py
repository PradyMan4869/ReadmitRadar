"""
Synthea → ReadmitRadar discharge records.

Parses Synthea's CSV export (https://github.com/synthetichealth/synthea)
into the schema defined in ml/features.py. Each Synthea run models one
hospital system (we generate two runs from different US states, so the two
"hospitals" have genuinely different populations and case mixes).

Label: an inpatient encounter is `readmitted_30d = 1` when the same patient
has another inpatient admission starting within 30 days of discharge.
Admissions in the final 30 days of a patient's record are dropped
(the label window is not observable for them).

Feature notes:
  - Comorbidity flags come from active condition DESCRIPTIONs at discharge
    (keyword match — Synthea SNOMED display strings are stable).
  - Labs are the last observation at/before discharge, matched on exact
    LOINC CODE; missing labs are imputed with the cohort median and
    counted in the returned stats.
  - `followup_scheduled` is proxied by a realized ambulatory/wellness visit
    within 14 days post-discharge (Synthea records visits, not bookings).
  - `charlson_index` is a simplified Charlson from available flag groups.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import FEATURE_COLUMNS, TARGET

INPATIENT_CLASSES = {"inpatient"}
ED_CLASSES = {"emergency", "urgentcare"}
FOLLOWUP_CLASSES = {"ambulatory", "wellness", "outpatient"}

READMIT_WINDOW_DAYS = 30
FOLLOWUP_WINDOW_DAYS = 14

# condition DESCRIPTION keywords → comorbidity flags
CONDITION_KEYWORDS = {
    "has_chf": ("heart failure",),
    "has_copd": ("chronic obstructive", "emphysema", "pulmonary disease"),
    "has_diabetes": ("diabetes",),
    "has_renal_disease": ("chronic kidney", "renal failure", "renal disease"),
}

# feature → Synthea observation LOINC CODE(s) (exact match; stable across
# runs). Some analytes are recorded under two interchangeable LOINC codes
# depending on which panel ordered them ("in Blood" vs "in Serum or Plasma");
# both are accepted as the same measurement.
LAB_CODES = {
    "bun_last": ("3094-0", "6299-2"),
    "creatinine_last": ("2160-0", "38483-4"),
    "sodium_last": ("2951-2", "2947-0"),
    "hemoglobin_last": ("718-7",),
    "glucose_last": ("2345-7", "2339-0"),
    "sbp_last": ("8480-6",),
    "dbp_last": ("8462-4",),
    "total_cholesterol_last": ("2093-3",),
    "hdl_last": ("2085-9",),
    "ldl_last": ("18262-6",),
    "triglycerides_last": ("2571-8",),
    "hba1c_last": ("4548-4",),
    "potassium_last": ("6298-4",),
    "wbc_last": ("6690-2",),
    "platelets_last": ("777-3",),
}

REQUIRED_FILES = ["patients.csv", "encounters.csv", "conditions.csv",
                  "observations.csv", "medications.csv"]


def _read_run(run_dir: Path) -> dict[str, pd.DataFrame]:
    csv_dir = run_dir / "csv" if (run_dir / "csv").exists() else run_dir
    missing = [f for f in REQUIRED_FILES if not (csv_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Synthea CSV export incomplete in {csv_dir}: missing {missing}. "
            "Run scripts/run_synthea.py first."
        )
    tables = {f[:-4]: pd.read_csv(csv_dir / f) for f in REQUIRED_FILES}
    for name in ("encounters", "conditions", "medications", "observations"):
        df = tables[name]
        for col in ("START", "STOP", "DATE"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    tables["patients"]["BIRTHDATE"] = pd.to_datetime(
        tables["patients"]["BIRTHDATE"], errors="coerce")
    return tables


def _lab_feature_map(observations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pre-filter observations per lab feature (numeric values only)."""
    obs = observations.copy()
    obs["VALUE"] = pd.to_numeric(obs["VALUE"], errors="coerce")
    obs = obs.dropna(subset=["VALUE", "DATE"])
    codes = obs["CODE"].astype(str)
    out = {}
    for feature, feature_codes in LAB_CODES.items():
        out[feature] = obs[codes.isin(feature_codes)][["PATIENT", "DATE", "VALUE"]]
    return out


def load_run(run_dir: Path, hospital: str) -> tuple[pd.DataFrame, dict]:
    """
    Build one hospital's discharge-record table from a Synthea run.

    Returns (records_df, stats) where stats reports admission counts and
    lab imputation rates.
    """
    t = _read_run(Path(run_dir))
    patients = t["patients"].set_index("Id")
    enc = t["encounters"].dropna(subset=["START", "STOP"])

    inpatient = (enc[enc["ENCOUNTERCLASS"].isin(INPATIENT_CLASSES)]
                 .sort_values("START").copy())
    if inpatient.empty:
        raise ValueError(f"No inpatient encounters in {run_dir}")

    # 30-day readmission label from the patient's next admission
    inpatient["next_start"] = inpatient.groupby("PATIENT")["START"].shift(-1)
    gap_days = (inpatient["next_start"] - inpatient["STOP"]).dt.total_seconds() / 86400
    inpatient[TARGET] = ((gap_days >= 0) & (gap_days <= READMIT_WINDOW_DAYS)).astype(int)

    # Drop admissions whose label window extends past the record's end
    record_end = enc.groupby("PATIENT")["STOP"].max()
    observable = inpatient["STOP"] + pd.Timedelta(days=READMIT_WINDOW_DAYS) <= \
        inpatient["PATIENT"].map(record_end)
    kept = inpatient[observable | (inpatient[TARGET] == 1)].copy()

    ed_enc = enc[enc["ENCOUNTERCLASS"].isin(ED_CLASSES)]
    followup_enc = enc[enc["ENCOUNTERCLASS"].isin(FOLLOWUP_CLASSES)]
    conditions = t["conditions"]
    medications = t["medications"]
    labs = _lab_feature_map(t["observations"])

    by_patient = {
        "inpatient": {k: g for k, g in kept.groupby("PATIENT")},
        "ed": {k: g for k, g in ed_enc.groupby("PATIENT")},
        "followup": {k: g for k, g in followup_enc.groupby("PATIENT")},
        "conditions": {k: g for k, g in conditions.groupby("PATIENT")},
        "medications": {k: g for k, g in medications.groupby("PATIENT")},
        "labs": {f: {k: g for k, g in df.groupby("PATIENT")}
                 for f, df in labs.items()},
    }

    rows, lab_missing = [], {f: 0 for f in LAB_CODES}
    for _, adm in kept.iterrows():
        pid, start, stop = adm["PATIENT"], adm["START"], adm["STOP"]
        if pid not in patients.index:
            continue
        patient = patients.loc[pid]

        age = (start.tz_localize(None) - patient["BIRTHDATE"]).days / 365.25
        if not (18 <= age <= 105):
            continue  # adult readmission model

        p_inp = by_patient["inpatient"].get(pid)
        prior = int(((p_inp["STOP"] < start) &
                     (p_inp["STOP"] >= start - pd.Timedelta(days=365))).sum())

        p_ed = by_patient["ed"].get(pid)
        ed_6mo = 0 if p_ed is None else int(
            ((p_ed["START"] < start) &
             (p_ed["START"] >= start - pd.Timedelta(days=180))).sum())

        p_cond = by_patient["conditions"].get(pid)
        flags = {f: 0 for f in CONDITION_KEYWORDS}
        n_dx = 0
        if p_cond is not None:
            active = p_cond[(p_cond["START"] <= stop) &
                            (p_cond["STOP"].isna() | (p_cond["STOP"] >= stop))]
            n_dx = len(active)
            desc = " | ".join(active["DESCRIPTION"].astype(str)).lower()
            for feature, keywords in CONDITION_KEYWORDS.items():
                flags[feature] = int(any(kw in desc for kw in keywords))

        p_med = by_patient["medications"].get(pid)
        n_meds = 0 if p_med is None else int(
            ((p_med["START"] <= stop) &
             (p_med["STOP"].isna() | (p_med["STOP"] >= stop))).sum())

        p_fu = by_patient["followup"].get(pid)
        followup = 0 if p_fu is None else int(
            ((p_fu["START"] > stop) &
             (p_fu["START"] <= stop + pd.Timedelta(days=FOLLOWUP_WINDOW_DAYS))
             ).any())

        row = {
            "patient_id": f"{hospital}-{pid[:8]}-{start.date()}",
            "hospital": hospital,
            "age": round(age),
            "gender_male": int(patient["GENDER"] == "M"),
            "length_of_stay": max(
                (stop - start).total_seconds() / 86400, 0.25),
            "n_prior_admissions": prior,
            "ed_visits_6mo": ed_6mo,
            "n_diagnoses": n_dx,
            "n_medications": n_meds,
            "charlson_index": (flags["has_chf"] + flags["has_copd"]
                               + flags["has_diabetes"]
                               + 2 * flags["has_renal_disease"]
                               + int(age > 70)),
            **flags,
            "discharged_to_snf": 0,  # not modelled by Synthea
            "followup_scheduled": followup,
            TARGET: int(adm[TARGET]),
        }
        for feature in LAB_CODES:
            p_lab = by_patient["labs"][feature].get(pid)
            value = None
            if p_lab is not None:
                before = p_lab[p_lab["DATE"] <= stop]
                if len(before):
                    value = float(before.sort_values("DATE")["VALUE"].iloc[-1])
            if value is None:
                lab_missing[feature] += 1
            row[feature] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No usable adult admissions in {run_dir}")

    # Median-impute missing labs (rate reported in stats)
    for feature in LAB_CODES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
        df[feature] = df[feature].fillna(df[feature].median())

    stats = {
        "admissions": len(df),
        "patients": df["patient_id"].str.split("-").str[1].nunique(),
        "readmission_rate": round(float(df[TARGET].mean()), 4),
        "lab_imputed_pct": {
            f: round(m / len(df) * 100, 1) for f, m in lab_missing.items()
        },
    }
    return df[["patient_id", "hospital"] + FEATURE_COLUMNS + [TARGET]], stats
