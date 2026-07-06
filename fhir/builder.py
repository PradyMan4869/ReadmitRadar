"""Build a FHIR R4 Bundle (JSON dict) from one discharge feature row."""
from ml.features import FEATURE_UNITS

from .resources import (
    CONDITION_CODES, ICD10_SYSTEM, LOCAL_CODES, LOCAL_SYSTEM, LOINC_CODES,
    LOINC_SYSTEM, SNF_DISPOSITION_CODE,
)


def _patient_entry(pid: str, row: dict) -> dict:
    return {"resource": {
        "resourceType": "Patient",
        "id": pid,
        "gender": "male" if int(row["gender_male"]) == 1 else "female",
        # Age is carried as an extension: the model consumes age-at-discharge,
        # and synthetic records have no meaningful absolute birth date.
        "extension": [{
            "url": "urn:readmitradar:age-years",
            "valueInteger": int(row["age"]),
        }],
    }}


def _encounter_entry(pid: str, row: dict) -> dict:
    resource = {
        "resourceType": "Encounter",
        "id": f"{pid}-enc",
        "status": "finished",
        "class": {"code": "IMP", "display": "inpatient encounter"},
        "subject": {"reference": f"Patient/{pid}"},
        "length": {"value": float(row["length_of_stay"]), "unit": "d"},
    }
    if int(row["discharged_to_snf"]) == 1:
        resource["hospitalization"] = {"dischargeDisposition": {"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/discharge-disposition",
            "code": SNF_DISPOSITION_CODE,
            "display": "Skilled nursing facility",
        }]}}
    return {"resource": resource}


def _condition_entries(pid: str, row: dict) -> list:
    entries = []
    for feature, (code, display) in CONDITION_CODES.items():
        if int(row[feature]) == 1:
            entries.append({"resource": {
                "resourceType": "Condition",
                "id": f"{pid}-{feature}",
                "subject": {"reference": f"Patient/{pid}"},
                "clinicalStatus": {"coding": [{"code": "active"}]},
                "code": {"coding": [{
                    "system": ICD10_SYSTEM, "code": code, "display": display,
                }]},
            }})
    return entries


def _observation_entries(pid: str, row: dict) -> list:
    entries = []
    for feature, (code, display) in LOINC_CODES.items():
        entries.append(_observation(pid, feature, code, display,
                                    LOINC_SYSTEM, row))
    for feature, (code, display) in LOCAL_CODES.items():
        entries.append(_observation(pid, feature, code, display,
                                    LOCAL_SYSTEM, row))
    return entries


def _observation(pid, feature, code, display, system, row) -> dict:
    return {"resource": {
        "resourceType": "Observation",
        "id": f"{pid}-{feature}",
        "status": "final",
        "subject": {"reference": f"Patient/{pid}"},
        "code": {"coding": [{"system": system, "code": code, "display": display}]},
        "valueQuantity": {
            "value": float(row[feature]),
            "unit": FEATURE_UNITS.get(feature, ""),
        },
    }}


def build_bundle(row: dict, patient_id: str = None) -> dict:
    """
    Convert one feature row (schema: ml.features.FEATURE_COLUMNS) into a
    FHIR R4 Bundle of type 'collection'.
    """
    pid = patient_id or str(row.get("patient_id", "unknown"))
    entries = [_patient_entry(pid, row), _encounter_entry(pid, row)]
    entries += _condition_entries(pid, row)
    entries += _observation_entries(pid, row)
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "id": f"discharge-{pid}",
        "entry": entries,
    }
