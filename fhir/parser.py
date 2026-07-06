"""
Parse a FHIR R4 Bundle into a model feature row.

This is the system's actual input path: the UI and the demo script accept a
Bundle, run it through here, and feed the result to the model. Unknown
resource types are ignored; missing features are reported explicitly rather
than silently imputed.
"""
from ml.features import FEATURE_COLUMNS, validate_row

from .resources import (
    CONDITION_CODES, LOCAL_CODES, LOINC_CODES, SNF_DISPOSITION_CODE,
)

# code → feature reverse maps
_LOINC_TO_FEATURE = {code: f for f, (code, _) in LOINC_CODES.items()}
_LOCAL_TO_FEATURE = {code: f for f, (code, _) in LOCAL_CODES.items()}
_ICD10_TO_FEATURE = {code: f for f, (code, _) in CONDITION_CODES.items()}


class BundleParseError(ValueError):
    """Raised when a Bundle cannot be turned into a complete feature row."""


def _parse_patient(resource: dict, row: dict) -> None:
    row["gender_male"] = 1 if resource.get("gender") == "male" else 0
    for ext in resource.get("extension", []):
        if ext.get("url") == "urn:readmitradar:age-years":
            row["age"] = float(ext.get("valueInteger", 0))


def _parse_encounter(resource: dict, row: dict) -> None:
    length = resource.get("length", {})
    if "value" in length:
        row["length_of_stay"] = float(length["value"])
    disposition = (
        resource.get("hospitalization", {})
        .get("dischargeDisposition", {})
        .get("coding", [{}])
    )
    codes = {c.get("code") for c in disposition}
    row["discharged_to_snf"] = 1 if SNF_DISPOSITION_CODE in codes else 0


def _parse_condition(resource: dict, row: dict) -> None:
    for coding in resource.get("code", {}).get("coding", []):
        feature = _ICD10_TO_FEATURE.get(coding.get("code"))
        if feature:
            row[feature] = 1


def _parse_observation(resource: dict, row: dict) -> None:
    value = resource.get("valueQuantity", {}).get("value")
    if value is None:
        return
    for coding in resource.get("code", {}).get("coding", []):
        feature = (_LOINC_TO_FEATURE.get(coding.get("code"))
                   or _LOCAL_TO_FEATURE.get(coding.get("code")))
        if feature:
            row[feature] = float(value)


_PARSERS = {
    "Patient": _parse_patient,
    "Encounter": _parse_encounter,
    "Condition": _parse_condition,
    "Observation": _parse_observation,
}


def parse_bundle(bundle: dict) -> dict:
    """
    Extract a complete feature row from a FHIR R4 Bundle.

    Raises BundleParseError listing every problem (wrong resourceType,
    missing features, invalid values) instead of failing one at a time.
    """
    if bundle.get("resourceType") != "Bundle":
        raise BundleParseError(
            f"expected resourceType 'Bundle', got {bundle.get('resourceType')!r}"
        )

    # Absent conditions/SNF flag legitimately mean 0; everything else must
    # be present in the Bundle.
    row = {f: 0 for f in list(CONDITION_CODES) + ["discharged_to_snf"]}
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        parser = _PARSERS.get(resource.get("resourceType"))
        if parser:
            parser(resource, row)

    problems = validate_row(row)
    if problems:
        raise BundleParseError("; ".join(problems))
    return {f: row[f] for f in FEATURE_COLUMNS}
