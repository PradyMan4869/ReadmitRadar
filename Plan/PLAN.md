# ReadmitRadar — Implementation Plan

**Goal:** Predict 30-day hospital readmission risk at discharge, with a fully on-prem,
HIPAA-conscious architecture: FHIR R4 input, federated training across two simulated
hospitals, SHAP explanations, local-LLM clinical rationale, AutoGen deliberation,
Langfuse tracing, Panel UI.

**Dataset (updated):** MIMIC-III credentialing was not obtainable, so patient data
comes from **Synthea** (synthetichealth/synthea): two independent runs (Massachusetts
/ Texas) model two hospital systems. `data/synthea_loader.py` derives 30-day
readmission labels from encounter sequences and emits the shared schema.
`data/synthetic.py` remains as the fast generator backing the unit tests.

---

## Architecture

```
FHIR R4 Bundle ──parse──► feature row ──► federated XGBoost ──► risk score
                                              │                    │
                                              ▼                    ▼
                                        SHAP values ──► LM Studio note
                                                            │
                                                            ▼
                                        AutoGen deliberation (Clinician ⇄ Risk Analyst)
                                                            │
                                                            ▼
                                                  Panel dashboard
   (every LLM step traced in self-hosted Langfuse; no PHI leaves the environment)
```

## Modules

| Module | Responsibility |
|---|---|
| `data/synthetic.py` | Synthetic discharge records; Hospital A (older, cardiac-heavy) vs Hospital B (younger, metabolic-heavy) case mixes so cross-hospital generalisation is a real effect |
| `data/synthea_loader.py` | Parse Synthea CSV export → discharge records with 30-day readmission labels (same schema) |
| `scripts/run_synthea.py` / `scripts/prepare_synthea.py` | Run Synthea for both hospitals; build train/test CSVs (patient-level split) |
| `fhir/resources.py` | Minimal FHIR R4 Patient / Encounter / Condition / Observation dataclasses with validation |
| `fhir/builder.py` | Tabular record → FHIR R4 Bundle (JSON) |
| `fhir/parser.py` | FHIR Bundle → model feature row (the system's actual input path) |
| `ml/features.py` | Feature schema — single source of truth for column names/types |
| `ml/model.py` | XGBoost readmission classifier wrapper (train/predict/save/load) |
| `ml/federated.py` | Hospital split; three regimes: local-only, centralized (privacy-violating upper bound), federated. Federated = FedAvg on a logistic model (true weight averaging, multi-round) **and** tree-adapted aggregation for XGBoost (margin averaging across hospital boosters — trees have no dense weight vector; documented honestly) |
| `ml/explain.py` | SHAP TreeExplainer per prediction; top-k signed contributions |
| `llm/lmstudio_client.py` | OpenAI-compatible client for LM Studio; deterministic template fallback when LM Studio is offline |
| `llm/clinical_note.py` | SHAP output → plain-English clinical rationale prompt |
| `llm/deliberation.py` | AutoGen two-agent chat (Clinician vs Risk Analyst); manual two-turn fallback if `autogen` absent |
| `observability/tracing.py` | Langfuse wrapper; no-op when keys unset (so nothing ever blocks on it) |
| `scripts/generate_data.py` | Write synthetic hospital A/B/test CSVs |
| `scripts/train.py` | Train all three regimes, save models + metrics JSON + comparison chart |
| `scripts/predict_demo.py` | End-to-end demo: FHIR bundle → risk → SHAP → note → deliberation |
| `ui/app.py` | Panel dashboard: FHIR input, risk gauge, regime comparison, SHAP waterfall, deliberation transcript, clinical note |
| `tests/` | Pure-python unit tests: FHIR round-trip, feature schema, federated math, SHAP shapes. No network, no LLM required |

## Design decisions

1. **Everything degrades gracefully offline.** LM Studio down → templated note clearly
   marked as fallback. Langfuse unset → no-op tracer. AutoGen missing → manual loop.
   The ML pipeline never depends on the LLM layer.
2. **FHIR is the real input path**, not decoration: the UI and demo parse a Bundle into
   features via `fhir/parser.py`. README documents every consumed resource type.
3. **FedAvg honesty.** True FedAvg (client weight averaging over rounds) is shown on a
   logistic model; XGBoost federation uses margin-averaged per-hospital boosters. The
   README states why (trees ≠ dense weights) — this is an interview talking point, not
   a hidden hack.
4. **Synthea-first data strategy** — clinically realistic patients with zero
   credentialing; the fast in-repo generator keeps unit tests free of Java
   and dataset dependencies.

## Milestones

1. Scaffolding + config + synthetic data ✅ runnable
2. FHIR layer + tests
3. Federated training + metrics + chart
4. SHAP + LLM note + deliberation + tracing
5. Panel UI
6. README (FHIR Resource Catalog, HIPAA Architecture Decisions), end-to-end verify
