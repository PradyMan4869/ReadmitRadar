"""
Minimal FHIR R4 resource models.

Only the fields ReadmitRadar consumes are modelled — this is a typed,
validated projection of FHIR R4, not a full implementation. The README's
FHIR Resource Catalog documents exactly what is read from each resource.
"""
from dataclasses import dataclass, field
from typing import Optional

LOINC_SYSTEM = "http://loinc.org"
ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10-cm"
LOCAL_SYSTEM = "urn:readmitradar:measure"  # utilization/derived measures

# Lab observations (LOINC)
LOINC_CODES = {
    "bun_last":        ("3094-0",  "Urea nitrogen [Mass/volume] in Serum or Plasma"),
    "creatinine_last": ("2160-0",  "Creatinine [Mass/volume] in Serum or Plasma"),
    "sodium_last":     ("2951-2",  "Sodium [Moles/volume] in Serum or Plasma"),
    "hemoglobin_last": ("718-7",   "Hemoglobin [Mass/volume] in Blood"),
    "glucose_last":    ("2345-7",  "Glucose [Mass/volume] in Serum or Plasma"),
    "sbp_last":        ("8480-6",  "Systolic Blood Pressure"),
    "dbp_last":        ("8462-4",  "Diastolic Blood Pressure"),
    "total_cholesterol_last": ("2093-3", "Cholesterol [Mass/volume] in Serum or Plasma"),
    "hdl_last":        ("2085-9",  "Cholesterol in HDL [Mass/volume] in Serum or Plasma"),
    "ldl_last":        ("18262-6", "Cholesterol in LDL [Mass/volume] in Serum or Plasma by Direct assay"),
    "triglycerides_last": ("2571-8", "Triglyceride [Mass/volume] in Serum or Plasma"),
    "hba1c_last":      ("4548-4",  "Hemoglobin A1c/Hemoglobin.total in Blood"),
    "potassium_last":  ("6298-4",  "Potassium [Moles/volume] in Blood"),
    "wbc_last":        ("6690-2",  "Leukocytes [#/volume] in Blood by Automated count"),
    "platelets_last":  ("777-3",   "Platelets [#/volume] in Blood by Automated count"),
}

# Utilization / derived observations (local code system)
LOCAL_CODES = {
    "n_prior_admissions": ("prior-admissions-12mo", "Inpatient admissions, prior 12 months"),
    "ed_visits_6mo":      ("ed-visits-6mo",         "ED visits, prior 6 months"),
    "n_diagnoses":        ("active-diagnoses",       "Active diagnosis count"),
    "n_medications":      ("discharge-medications",  "Discharge medication count"),
    "charlson_index":     ("charlson-index",         "Charlson comorbidity index"),
    "followup_scheduled": ("followup-scheduled",     "Follow-up visit scheduled (1=yes)"),
}

# Comorbidity conditions (ICD-10-CM)
CONDITION_CODES = {
    "has_chf":           ("I50.9",  "Heart failure, unspecified"),
    "has_copd":          ("J44.9",  "Chronic obstructive pulmonary disease, unspecified"),
    "has_diabetes":      ("E11.9",  "Type 2 diabetes mellitus without complications"),
    "has_renal_disease": ("N18.9",  "Chronic kidney disease, unspecified"),
}

SNF_DISPOSITION_CODE = "snf"  # http://terminology.hl7.org/CodeSystem/discharge-disposition


@dataclass
class Patient:
    id: str
    gender: str            # "male" | "female"
    age_years: int

    def validate(self) -> list:
        problems = []
        if self.gender not in ("male", "female"):
            problems.append(f"Patient.gender must be male/female, got {self.gender!r}")
        if not (0 <= self.age_years <= 120):
            problems.append(f"Patient age out of range: {self.age_years}")
        return problems


@dataclass
class Encounter:
    id: str
    patient_id: str
    length_of_stay_days: float
    discharge_disposition: Optional[str] = None  # e.g. "snf", "home"

    def validate(self) -> list:
        problems = []
        if self.length_of_stay_days < 0:
            problems.append(f"Encounter LOS negative: {self.length_of_stay_days}")
        return problems


@dataclass
class Condition:
    id: str
    patient_id: str
    code: str              # ICD-10-CM
    display: str
    system: str = ICD10_SYSTEM

    def validate(self) -> list:
        return [] if self.code else ["Condition.code missing"]


@dataclass
class Observation:
    id: str
    patient_id: str
    code: str
    display: str
    value: float
    unit: str = ""
    system: str = LOINC_SYSTEM

    def validate(self) -> list:
        problems = []
        if not self.code:
            problems.append("Observation.code missing")
        try:
            float(self.value)
        except (TypeError, ValueError):
            problems.append(f"Observation.value not numeric: {self.value!r}")
        return problems
