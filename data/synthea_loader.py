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
  - `charlson_index` is a simplified Charlson from available flag groups.
  - There is deliberately no "follow-up scheduled" feature: Synthea only
    records realized visits, not bookings made at discharge time, so the
    only available proxy (a visit actually occurring in the following
    weeks) is information that doesn't exist yet at the moment the model
    is meant to score — a real temporal-leakage risk, not a usable signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import FEATURE_COLUMNS, TARGET

INPATIENT_CLASSES = {"inpatient"}
ED_CLASSES = {"emergency", "urgentcare"}

READMIT_WINDOW_DAYS = 30

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


def _asof_last_value(admissions: pd.DataFrame, events: pd.DataFrame,
                      value_col: str) -> pd.Series:
    """
    For each admission (PATIENT, STOP), the most recent `events` row for
    that patient with DATE <= STOP — a vectorized point-in-time join
    replacing a per-admission "filter then take the last" loop.
    """
    if events.empty:
        return pd.Series(np.nan, index=admissions.index)
    ev = events.sort_values("DATE", kind="stable")
    adm = admissions[["PATIENT", "STOP"]].reset_index().sort_values("STOP", kind="stable")
    merged = pd.merge_asof(
        adm, ev, left_on="STOP", right_on="DATE",
        by="PATIENT", direction="backward",
    )
    return merged.set_index("index")[value_col].reindex(admissions.index)


def _active_interval_count(admissions: pd.DataFrame, anchor_col: str,
                            intervals: pd.DataFrame) -> pd.Series:
    """
    For each admission, count same-patient `intervals` (columns PATIENT,
    START, STOP — STOP may be NaT for "still active") active at the
    admission's anchor time: `START <= anchor <= STOP` (or STOP is NaT).

    Vectorized as "started on/before anchor" minus "already ended before
    anchor" — two merge_asof running counts per side, same pattern as
    _windowed_count, so no per-admission cross join of every interval is
    ever materialized (that approach blows up combinatorially at realistic
    hospital population sizes: every admission x every lifetime condition).
    """
    if intervals.empty:
        return pd.Series(0, index=admissions.index)

    adm = admissions[["PATIENT", anchor_col]].reset_index()

    starts = intervals[["PATIENT", "START"]].sort_values("START", kind="stable").copy()
    starts["_cum"] = starts.groupby("PATIENT").cumcount() + 1
    started_sorted = adm.sort_values(anchor_col)
    merged_started = pd.merge_asof(
        started_sorted, starts, left_on=anchor_col, right_on="START",
        by="PATIENT", direction="backward", allow_exact_matches=True,
    )
    n_started = merged_started.set_index("index")["_cum"].reindex(adm["index"]).fillna(0)

    ended = intervals.dropna(subset=["STOP"])[["PATIENT", "STOP"]].sort_values(
        "STOP", kind="stable").copy()
    if ended.empty:
        n_ended = pd.Series(0, index=adm["index"])
    else:
        ended["_cum"] = ended.groupby("PATIENT").cumcount() + 1
        ended_sorted = adm.sort_values(anchor_col)
        merged_ended = pd.merge_asof(
            ended_sorted, ended, left_on=anchor_col, right_on="STOP",
            by="PATIENT", direction="backward", allow_exact_matches=False,
        )
        n_ended = merged_ended.set_index("index")["_cum"].reindex(adm["index"]).fillna(0)

    result = n_started - n_ended
    result.index = admissions.index
    return result.astype(int)


def _windowed_count(admissions: pd.DataFrame, anchor_col: str,
                     events: pd.DataFrame, event_time_col: str,
                     window_days: float) -> pd.Series:
    """
    For each admission, count same-patient `events` with
    `admissions[anchor_col] - window_days <= events[event_time_col] < admissions[anchor_col]`
    — vectorized via two merge_asof point-in-time counts (events strictly
    before the anchor, minus events strictly before the window start).
    `events` must exclude the admissions table itself when counting prior
    admissions, so a same-day second admission never counts itself.
    """
    if events.empty:
        return pd.Series(0, index=admissions.index)

    ev = events[["PATIENT", event_time_col]].sort_values(event_time_col).copy()
    ev["_cum"] = ev.groupby("PATIENT").cumcount() + 1

    adm = admissions[["PATIENT", anchor_col]].reset_index()

    # Count of events strictly before the admission anchor time (exclusive,
    # so an event at the exact same instant as the anchor doesn't count —
    # e.g. two admissions starting at the same timestamp).
    before_sorted = adm.sort_values(anchor_col)
    merged_before = pd.merge_asof(
        before_sorted, ev, left_on=anchor_col, right_on=event_time_col,
        by="PATIENT", direction="backward", allow_exact_matches=False,
    )
    count_before = merged_before.set_index("index")["_cum"].reindex(adm["index"]).fillna(0)

    # Count of events strictly before the window start (anchor - window)
    adm["_window_start"] = adm[anchor_col] - pd.Timedelta(days=window_days)
    window_sorted = adm.sort_values("_window_start")
    merged_window = pd.merge_asof(
        window_sorted, ev, left_on="_window_start", right_on=event_time_col,
        by="PATIENT", direction="backward", allow_exact_matches=False,
    )
    count_before_window = merged_window.set_index("index")["_cum"].reindex(adm["index"]).fillna(0)

    result = count_before - count_before_window
    result.index = admissions.index
    return result.astype(int)


def load_run(run_dir: Path, hospital: str) -> tuple[pd.DataFrame, dict]:
    """
    Build one hospital's discharge-record table from a Synthea run.

    Returns (records_df, stats) where stats reports admission counts and
    lab imputation rates. Vectorized (merge_asof point-in-time joins +
    windowed counts) rather than a per-admission Python loop, since a
    realistic hospital population (tens of thousands of admissions) makes
    a 15-lab-lookups-per-row loop the dominant cost.
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
    kept = (inpatient[observable | (inpatient[TARGET] == 1)]
            .merge(patients[["BIRTHDATE", "GENDER"]], left_on="PATIENT",
                   right_index=True, how="left")
            .reset_index(drop=True))

    age_years = (kept["START"].dt.tz_localize(None) - kept["BIRTHDATE"]).dt.days / 365.25
    kept = kept[(age_years >= 18) & (age_years <= 105)].copy()
    kept["_age_exact"] = age_years[kept.index]
    kept["age"] = kept["_age_exact"].round()
    kept["gender_male"] = (kept["GENDER"] == "M").astype(int)
    kept["length_of_stay"] = ((kept["STOP"] - kept["START"]).dt.total_seconds()
                               / 86400).clip(lower=0.25)
    kept["patient_id"] = (hospital + "-" + kept["PATIENT"].str[:8] + "-"
                           + kept["START"].dt.date.astype(str))

    ed_enc = enc[enc["ENCOUNTERCLASS"].isin(ED_CLASSES)]
    conditions = t["conditions"]
    medications = t["medications"]
    labs = _lab_feature_map(t["observations"])

    kept["n_prior_admissions"] = _windowed_count(
        kept, "START", kept[["PATIENT", "STOP"]], "STOP", 365)
    kept["ed_visits_6mo"] = _windowed_count(
        kept, "START", ed_enc[["PATIENT", "START"]], "START", 180)

    # n_diagnoses: count of conditions active (STARTed on/before discharge,
    # not yet STOPped) at discharge time — via the same active-interval
    # count used for the comorbidity flags below, over every condition.
    kept["n_diagnoses"] = _active_interval_count(kept, "STOP", conditions)

    desc_lower = conditions["DESCRIPTION"].astype(str).str.lower()
    for feature, keywords in CONDITION_KEYWORDS.items():
        matching = conditions[desc_lower.str.contains("|".join(keywords))]
        kept[feature] = (_active_interval_count(kept, "STOP", matching) > 0).astype(int)

    kept["n_medications"] = _active_interval_count(kept, "STOP", medications)

    kept["charlson_index"] = (
        kept["has_chf"] + kept["has_copd"] + kept["has_diabetes"]
        + 2 * kept["has_renal_disease"] + (kept["_age_exact"] > 70).astype(int)
    )
    kept["discharged_to_snf"] = 0  # not modelled by Synthea

    lab_missing = {}
    for feature in LAB_CODES:
        lab_events = labs[feature].rename(columns={"VALUE": "_value"})
        values = _asof_last_value(kept, lab_events, "_value")
        lab_missing[feature] = int(values.isna().sum())
        kept[feature] = pd.to_numeric(values, errors="coerce")

    df = kept[["patient_id", "PATIENT"] + FEATURE_COLUMNS + [TARGET]].copy()
    df.insert(1, "hospital", hospital)
    df = df.drop(columns=["PATIENT"])
    if df.empty:
        raise ValueError(f"No usable adult admissions in {run_dir}")

    # Median-impute missing labs (rate reported in stats)
    for feature in LAB_CODES:
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
