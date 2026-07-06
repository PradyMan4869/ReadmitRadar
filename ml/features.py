"""
Feature schema — the single source of truth for model inputs.

Every producer (Synthea loader, synthetic generator, FHIR parser) must emit
exactly these columns; every consumer (models, SHAP, UI) reads them from here.
"""
from dataclasses import dataclass

TARGET = "readmitted_30d"

# (name, human label, unit) — order defines the model's column order
FEATURES = [
    ("age",                  "Age",                          "years"),
    ("gender_male",          "Male gender",                  "0/1"),
    ("length_of_stay",       "Length of stay",               "days"),
    ("n_prior_admissions",   "Prior admissions (12 mo)",     "count"),
    ("ed_visits_6mo",        "ED visits (6 mo)",             "count"),
    ("n_diagnoses",          "Active diagnoses",             "count"),
    ("n_medications",        "Discharge medications",        "count"),
    ("charlson_index",       "Charlson comorbidity index",   "score"),
    ("has_chf",              "Congestive heart failure",     "0/1"),
    ("has_copd",             "COPD",                         "0/1"),
    ("has_diabetes",         "Diabetes mellitus",            "0/1"),
    ("has_renal_disease",    "Chronic kidney disease",       "0/1"),
    ("bun_last",             "BUN at discharge",             "mg/dL"),
    ("creatinine_last",      "Creatinine at discharge",      "mg/dL"),
    ("sodium_last",          "Sodium at discharge",          "mEq/L"),
    ("hemoglobin_last",      "Hemoglobin at discharge",      "g/dL"),
    ("glucose_last",         "Glucose at discharge",         "mg/dL"),
    ("sbp_last",             "Systolic BP at discharge",     "mmHg"),
    ("dbp_last",             "Diastolic BP at discharge",    "mmHg"),
    ("total_cholesterol_last", "Total cholesterol at discharge", "mg/dL"),
    ("hdl_last",             "HDL cholesterol at discharge", "mg/dL"),
    ("ldl_last",             "LDL cholesterol at discharge", "mg/dL"),
    ("triglycerides_last",   "Triglycerides at discharge",   "mg/dL"),
    ("hba1c_last",           "HbA1c at discharge",           "%"),
    ("potassium_last",       "Potassium at discharge",       "mmol/L"),
    ("wbc_last",             "WBC at discharge",             "10*3/uL"),
    ("platelets_last",       "Platelets at discharge",       "10*3/uL"),
    ("discharged_to_snf",    "Discharged to skilled nursing","0/1"),
    ("followup_scheduled",   "Follow-up visit scheduled",    "0/1"),
]

FEATURE_COLUMNS = [name for name, _, _ in FEATURES]
FEATURE_LABELS = {name: label for name, label, _ in FEATURES}
FEATURE_UNITS = {name: unit for name, _, unit in FEATURES}

BINARY_FEATURES = {n for n, _, u in FEATURES if u == "0/1"}
LAB_FEATURES = [n for n in FEATURE_COLUMNS if n.endswith("_last")]


@dataclass(frozen=True)
class RiskBands:
    """Cut-points used consistently by the UI and the deliberation agents."""
    low: float = 0.15
    moderate: float = 0.35
    high: float = 0.60

    def label(self, p: float) -> str:
        if p >= self.high:
            return "HIGH"
        if p >= self.moderate:
            return "ELEVATED"
        if p >= self.low:
            return "MODERATE"
        return "LOW"


RISK_BANDS = RiskBands()


def validate_row(row: dict) -> list:
    """Return a list of problems with a feature row (empty = valid)."""
    problems = []
    for col in FEATURE_COLUMNS:
        if col not in row or row[col] is None:
            problems.append(f"missing feature: {col}")
            continue
        try:
            v = float(row[col])
        except (TypeError, ValueError):
            problems.append(f"non-numeric value for {col}: {row[col]!r}")
            continue
        if col in BINARY_FEATURES and v not in (0.0, 1.0):
            problems.append(f"{col} must be 0/1, got {v}")
        if col == "age" and not (0 <= v <= 120):
            problems.append(f"age out of range: {v}")
    return problems
