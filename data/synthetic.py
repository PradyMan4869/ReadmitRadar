"""
Synthetic MIMIC-like discharge records.

Two hospitals with deliberately different case mixes so cross-hospital
generalisation is a real, measurable effect (not an artifact):

  Hospital A — tertiary cardiac centre: older patients, more CHF/CKD,
               longer stays, higher BUN/creatinine.
  Hospital B — urban community hospital: younger, metabolic-heavy
               (diabetes, glucose), more ED churn, shorter stays.

The 30-day readmission label is generated from a shared logistic ground
truth over clinically sensible drivers, plus noise. Because the *label
mechanism* is shared but the *covariate distributions* differ, a model
trained on one hospital degrades on the other — exactly the situation
federated learning addresses.

The production pipeline uses data/synthea_loader.py (same schema); this
generator remains as the fast, dependency-free backing for the unit tests.
"""
import numpy as np
import pandas as pd

from ml.features import FEATURE_COLUMNS, TARGET

# Ground-truth log-odds weights (clinically directionally correct)
_TRUE_WEIGHTS = {
    "age": 0.020,               # per year over 60
    "length_of_stay": 0.045,
    "n_prior_admissions": 0.38,
    "ed_visits_6mo": 0.22,
    "n_diagnoses": 0.06,
    "n_medications": 0.03,
    "charlson_index": 0.16,
    "has_chf": 0.55,
    "has_copd": 0.35,
    "has_diabetes": 0.20,
    "has_renal_disease": 0.45,
    "bun_last": 0.018,          # per mg/dL over 20
    "creatinine_last": 0.30,    # per mg/dL over 1.0
    "sodium_last": -0.05,       # per mEq/L over 135 (hyponatremia risk below)
    "hemoglobin_last": -0.12,   # per g/dL (anemia raises risk)
    "glucose_last": 0.004,      # per mg/dL over 120
    "sbp_last": 0.0025,         # per mmHg over 130
    "dbp_last": 0.0015,         # per mmHg over 80
    "total_cholesterol_last": 0.0006,  # per mg/dL over 200
    "hdl_last": -0.006,         # per mg/dL (higher HDL protective)
    "ldl_last": 0.0010,         # per mg/dL over 100
    "triglycerides_last": 0.0004,  # per mg/dL over 150
    "hba1c_last": 0.055,        # per % over 6.5
    "potassium_last": 0.04,     # per mEq/L over 4.5 (or under, symmetric risk)
    "wbc_last": 0.02,           # per 10^3/uL over 11 (infection signal)
    "platelets_last": -0.0012,  # per 10^3/uL under 200 (thrombocytopenia risk)
    "discharged_to_snf": 0.30,
    "followup_scheduled": -0.55,
    "gender_male": 0.05,
}
_INTERCEPT = -4.35  # calibrated to ~15-25% readmission rates per site


def _label_probability(df: pd.DataFrame) -> np.ndarray:
    z = np.full(len(df), _INTERCEPT)
    z += _TRUE_WEIGHTS["age"] * (df["age"] - 60)
    z += _TRUE_WEIGHTS["length_of_stay"] * df["length_of_stay"]
    z += _TRUE_WEIGHTS["n_prior_admissions"] * df["n_prior_admissions"]
    z += _TRUE_WEIGHTS["ed_visits_6mo"] * df["ed_visits_6mo"]
    z += _TRUE_WEIGHTS["n_diagnoses"] * df["n_diagnoses"]
    z += _TRUE_WEIGHTS["n_medications"] * df["n_medications"]
    z += _TRUE_WEIGHTS["charlson_index"] * df["charlson_index"]
    for flag in ("has_chf", "has_copd", "has_diabetes", "has_renal_disease",
                 "discharged_to_snf", "followup_scheduled", "gender_male"):
        z += _TRUE_WEIGHTS[flag] * df[flag]
    z += _TRUE_WEIGHTS["bun_last"] * (df["bun_last"] - 20)
    z += _TRUE_WEIGHTS["creatinine_last"] * (df["creatinine_last"] - 1.0)
    z += _TRUE_WEIGHTS["sodium_last"] * (df["sodium_last"] - 135)
    z += _TRUE_WEIGHTS["hemoglobin_last"] * (df["hemoglobin_last"] - 13)
    z += _TRUE_WEIGHTS["glucose_last"] * (df["glucose_last"] - 120)
    z += _TRUE_WEIGHTS["sbp_last"] * (df["sbp_last"] - 130)
    z += _TRUE_WEIGHTS["dbp_last"] * (df["dbp_last"] - 80)
    z += _TRUE_WEIGHTS["total_cholesterol_last"] * (df["total_cholesterol_last"] - 200)
    z += _TRUE_WEIGHTS["hdl_last"] * (df["hdl_last"] - 50)
    z += _TRUE_WEIGHTS["ldl_last"] * (df["ldl_last"] - 100)
    z += _TRUE_WEIGHTS["triglycerides_last"] * (df["triglycerides_last"] - 150)
    z += _TRUE_WEIGHTS["hba1c_last"] * (df["hba1c_last"] - 6.5)
    z += _TRUE_WEIGHTS["potassium_last"] * abs(df["potassium_last"] - 4.2)
    z += _TRUE_WEIGHTS["wbc_last"] * (df["wbc_last"] - 11)
    z += _TRUE_WEIGHTS["platelets_last"] * (200 - df["platelets_last"])
    return 1.0 / (1.0 + np.exp(-z))


def _clip(a, lo, hi):
    return np.clip(a, lo, hi)


def generate_hospital(hospital: str, n: int, seed: int) -> pd.DataFrame:
    """Generate `n` discharge records for hospital 'A' or 'B'."""
    if hospital not in ("A", "B"):
        raise ValueError(f"hospital must be 'A' or 'B', got {hospital!r}")
    rng = np.random.default_rng(seed)
    cardiac = hospital == "A"

    age = _clip(rng.normal(72 if cardiac else 58, 12, n), 18, 100).round()
    male = rng.binomial(1, 0.55 if cardiac else 0.48, n)
    los = _clip(rng.gamma(2.2, 2.6 if cardiac else 1.8, n), 1, 45).round()
    prior = rng.poisson(1.4 if cardiac else 0.9, n)
    ed = rng.poisson(0.7 if cardiac else 1.3, n)

    chf = rng.binomial(1, 0.42 if cardiac else 0.12, n)
    copd = rng.binomial(1, 0.22 if cardiac else 0.15, n)
    dm = rng.binomial(1, 0.28 if cardiac else 0.44, n)
    ckd = rng.binomial(1, 0.30 if cardiac else 0.13, n)

    ndx = _clip(rng.poisson(7 if cardiac else 5, n) + 2 * chf + dm, 1, 30)
    nmeds = _clip(rng.poisson(9 if cardiac else 7, n) + 2 * chf + ckd, 0, 40)
    charlson = _clip(chf + copd + dm + 2 * ckd + rng.poisson(1.2, n)
                     + (age > 70).astype(int), 0, 15)

    bun = _clip(rng.normal(24 if cardiac else 17, 9, n) + 14 * ckd, 4, 120).round(1)
    creat = _clip(rng.normal(1.25 if cardiac else 0.95, 0.4, n) + 1.1 * ckd, 0.3, 9).round(2)
    sodium = _clip(rng.normal(137, 3.4, n) - 1.6 * chf, 118, 152).round(1)
    hgb = _clip(rng.normal(12.1 if cardiac else 13.2, 1.9, n) - 1.0 * ckd, 5, 19).round(1)
    glucose = _clip(rng.normal(120 if cardiac else 138, 36, n) + 48 * dm, 55, 500).round()

    sbp = _clip(rng.normal(138 if cardiac else 128, 18, n) + 10 * chf, 80, 220).round()
    dbp = _clip(rng.normal(82 if cardiac else 78, 11, n), 40, 130).round()
    total_chol = _clip(rng.normal(195 if cardiac else 205, 38, n), 100, 400).round()
    hdl = _clip(rng.normal(42 if cardiac else 48, 12, n) - 4 * dm, 15, 100).round()
    ldl = _clip(rng.normal(115 if cardiac else 108, 32, n), 30, 260).round()
    trig = _clip(rng.normal(155 if cardiac else 165, 60, n) + 40 * dm, 40, 600).round()
    hba1c = _clip(rng.normal(6.0 if cardiac else 6.8, 1.1, n) + 1.6 * dm, 4.5, 14).round(1)
    potassium = _clip(rng.normal(4.3, 0.5, n) + 0.3 * ckd, 2.8, 6.5).round(1)
    wbc = _clip(rng.normal(8.5, 3.2, n), 2, 30).round(1)
    platelets = _clip(rng.normal(230 if cardiac else 250, 70, n), 20, 600).round()

    snf = rng.binomial(1, _clip(0.06 + 0.004 * (age - 60), 0.02, 0.55))
    followup = rng.binomial(1, 0.72 if cardiac else 0.58, n)

    df = pd.DataFrame({
        "age": age, "gender_male": male, "length_of_stay": los,
        "n_prior_admissions": prior, "ed_visits_6mo": ed,
        "n_diagnoses": ndx, "n_medications": nmeds, "charlson_index": charlson,
        "has_chf": chf, "has_copd": copd, "has_diabetes": dm,
        "has_renal_disease": ckd,
        "bun_last": bun, "creatinine_last": creat, "sodium_last": sodium,
        "hemoglobin_last": hgb, "glucose_last": glucose,
        "sbp_last": sbp, "dbp_last": dbp,
        "total_cholesterol_last": total_chol, "hdl_last": hdl, "ldl_last": ldl,
        "triglycerides_last": trig, "hba1c_last": hba1c,
        "potassium_last": potassium, "wbc_last": wbc, "platelets_last": platelets,
        "discharged_to_snf": snf, "followup_scheduled": followup,
    })
    df[TARGET] = rng.binomial(1, _label_probability(df))
    df.insert(0, "patient_id", [f"{hospital}-{i:05d}" for i in range(n)])
    df.insert(1, "hospital", hospital)
    return df[["patient_id", "hospital"] + FEATURE_COLUMNS + [TARGET]]


def generate_test_set(n: int, seed: int) -> pd.DataFrame:
    """Pooled held-out set: half from each hospital's distribution."""
    a = generate_hospital("A", n // 2, seed + 1000)
    b = generate_hospital("B", n - n // 2, seed + 2000)
    df = pd.concat([a, b], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
