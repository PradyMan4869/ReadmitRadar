"""End-to-end demo: FHIR Bundle → risk → SHAP → clinical note → deliberation.

Run after scripts/train.py:
    python scripts/predict_demo.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import DATA_DIR, MODELS_DIR, REPORTS_DIR
from fhir.builder import build_bundle
from fhir.parser import parse_bundle
from llm.clinical_note import generate_clinical_note
from llm.deliberation import deliberate
from ml.explain import RiskExplainer
from ml.features import RISK_BANDS
from ml.model import ReadmissionModel


def main() -> None:
    model_path = MODELS_DIR / "xgb_reference.json"
    if not model_path.exists():
        raise SystemExit("Run scripts/train.py first.")

    test = pd.read_csv(DATA_DIR / "test.csv")
    record = test.iloc[0].to_dict()

    # 1. Tabular record → FHIR R4 Bundle (what an EHR would send)
    bundle = build_bundle(record)
    (REPORTS_DIR / "sample_bundle.json").write_text(
        json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"FHIR Bundle: {len(bundle['entry'])} resources "
          f"({', '.join(sorted({e['resource']['resourceType'] for e in bundle['entry']}))})")

    # 2. Bundle → feature row (the system's real input path)
    row = parse_bundle(bundle)

    # 3. Risk + SHAP
    model = ReadmissionModel().load(model_path)
    risk = model.predict_row(row)
    contribs = RiskExplainer(model).explain_row(row)
    print(f"\nRisk: {risk:.1%} ({RISK_BANDS.label(risk)})")
    for c in contribs:
        print(f"  {c.label:<32} {c.shap:+.3f}")

    # 4. Clinical note (LM Studio, or clearly-marked template offline)
    note = generate_clinical_note(risk, contribs)
    print(f"\nClinical note [{note['source']}]:\n{note['note']}")

    # 5. Two-agent deliberation
    result = deliberate(risk, contribs)
    print(f"\nDeliberation [{result['source']}]:")
    for turn in result["transcript"]:
        print(f"  [{turn['role']}] {turn['content'][:200]}")


if __name__ == "__main__":
    main()
