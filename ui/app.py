"""
ReadmitRadar Panel dashboard.

Run after scripts/train.py:
    panel serve ui/app.py --show
(also works in Jupyter: `panel serve` is optional, the template renders inline)

Layout: sidebar navigation between
  - Overview: cohort-level risk distribution and headline metrics
  - Patients: sortable, risk-colored roster (held-out test set) — click a
    row to open that patient's detail (risk gauge, SHAP waterfall, clinical
    note, deliberation transcript)
  - Bundle input: paste an arbitrary FHIR R4 Bundle for ad-hoc scoring
Everything is computed locally; LM Studio and Langfuse are optional at
runtime — the pipeline degrades to deterministic templates without them.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import panel as pn

from config import DATA_DIR, MODELS_DIR, REPORTS_DIR
from fhir.builder import build_bundle
from fhir.parser import BundleParseError, parse_bundle
from llm.clinical_note import generate_clinical_note
from llm.deliberation import deliberate
from ml.explain import RiskExplainer
from ml.features import FEATURE_COLUMNS, FEATURE_LABELS, RISK_BANDS
from ml.model import ReadmissionModel

pn.extension("tabulator", sizing_mode="stretch_width")

ACCENT = "#2a78d6"
BAND_COLORS = {
    "LOW": "#0ca30c",
    "MODERATE": "#e8c209",
    "ELEVATED": "#e87f0e",
    "HIGH": "#d03b3b",
}


def _require(path: Path, hint: str):
    if not path.exists():
        raise SystemExit(f"{path} not found — {hint}")
    return path


MODEL = ReadmissionModel().load(
    _require(MODELS_DIR / "xgb_reference.json", "run scripts/train.py"))
EXPLAINER = RiskExplainer(MODEL)
TEST_DF = pd.read_csv(
    _require(DATA_DIR / "test.csv", "run scripts/generate_data.py"))

# ── Precompute the roster (risk score + band for every held-out patient) ────

_risk_scores = MODEL.predict_proba(TEST_DF)
ROSTER = TEST_DF[["patient_id", "hospital", "age", "gender_male",
                   "n_prior_admissions", "n_diagnoses", "length_of_stay",
                   "readmitted_30d"]].copy()
ROSTER.insert(2, "risk", _risk_scores)
ROSTER["risk_band"] = ROSTER["risk"].apply(RISK_BANDS.label)
ROSTER["gender"] = ROSTER["gender_male"].map({1: "M", 0: "F"})

# patient_id (patient + admission date) is NOT a unique key — 122 of 2902
# held-out rows share a patient_id with another admission of the same
# patient. row_id captures each row's position in TEST_DF *before* the
# risk-sort below, so it stays a valid TEST_DF.iloc[...] lookup key
# afterwards — using ROSTER's own post-sort positional index here was a
# bug: it silently resolved detail lookups to the wrong admission.
ROSTER["row_id"] = np.arange(len(ROSTER))

ROSTER = ROSTER.drop(columns=["gender_male"]).sort_values(
    "risk", ascending=False).reset_index(drop=True)

ROSTER["risk_pct"] = (ROSTER["risk"] * 100).round(1)
ROSTER["length_of_stay"] = ROSTER["length_of_stay"].round(1)

DISPLAY_COLUMNS = {
    "row_id": "row_id",
    "patient_id": "Patient",
    "hospital": "Site",
    "risk_pct": "Risk %",
    "risk_band": "Band",
    "age": "Age",
    "gender": "Sex",
    "n_prior_admissions": "Prior adm.",
    "n_diagnoses": "Diagnoses",
    "length_of_stay": "LOS (d)",
}
ROSTER_VIEW = ROSTER[list(DISPLAY_COLUMNS)].rename(columns=DISPLAY_COLUMNS)

# ── Overview stats ──────────────────────────────────────────────────────────

N_PATIENTS = len(ROSTER)
BAND_COUNTS = ROSTER["risk_band"].value_counts().reindex(
    ["LOW", "MODERATE", "ELEVATED", "HIGH"], fill_value=0)
PCT_HIGH_OR_ELEVATED = (BAND_COUNTS["HIGH"] + BAND_COUNTS["ELEVATED"]) / N_PATIENTS
MEAN_RISK = ROSTER["risk"].mean()
ACTUAL_READMIT_RATE = ROSTER["readmitted_30d"].mean()


def _stat_card(value: str, label: str, color: str = "#0b0b0b") -> pn.pane.HTML:
    return pn.pane.HTML(f"""
    <div style="padding:18px;border:1px solid #e3e1dc;border-radius:10px;
                text-align:center;background:#ffffff">
      <div style="font-size:30px;font-weight:700;color:{color}">{value}</div>
      <div style="font-size:13px;color:#52514e;margin-top:4px">{label}</div>
    </div>""", sizing_mode="stretch_width")


def _band_bar_html() -> str:
    total = max(N_PATIENTS, 1)
    segments = "".join(
        f'<div style="width:{BAND_COUNTS[b]/total*100:.1f}%;background:'
        f'{BAND_COLORS[b]};height:28px" title="{b}: {BAND_COUNTS[b]}"></div>'
        for b in ("LOW", "MODERATE", "ELEVATED", "HIGH")
    )
    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:16px">'
        f'<span style="width:10px;height:10px;background:{BAND_COLORS[b]};'
        f'border-radius:2px;display:inline-block;margin-right:6px"></span>'
        f'{b} ({BAND_COUNTS[b]})</span>'
        for b in ("LOW", "MODERATE", "ELEVATED", "HIGH")
    )
    return f"""
    <div style="display:flex;border-radius:6px;overflow:hidden;margin-bottom:10px">
      {segments}
    </div>
    <div style="font-size:12px;color:#52514e">{legend}</div>
    """


_METRICS_PATH = REPORTS_DIR / "federated_metrics.json"
_REGIME_METRICS = (json.loads(_METRICS_PATH.read_text(encoding="utf-8"))
                   if _METRICS_PATH.exists() else {})


def _regime_table_html() -> str:
    rows_spec = [
        ("local_A", "Local — Hospital A only",
         "Trained only on Hospital A's own patients. What Hospital A could "
         "do completely alone, with zero data sharing."),
        ("local_B", "Local — Hospital B only",
         "Trained only on Hospital B's own patients. What Hospital B could "
         "do completely alone."),
        ("local_A_on_other_site", "Hospital A's model, tested on Hospital B",
         "Diagnostic only — shows how much a single-site model degrades on "
         "a population it never saw. This gap is the whole reason "
         "federated learning is worth doing."),
        ("local_B_on_other_site", "Hospital B's model, tested on Hospital A",
         "Same diagnostic, reversed."),
        ("centralized", "Centralized (pooled data)",
         "Trained on both hospitals' raw patient records pooled together. "
         "The best possible score — but illegal under HIPAA without a "
         "data-sharing agreement, since it means moving real patient "
         "records between covered entities. Shown only as the upper-bound "
         "benchmark federated learning is measured against."),
        ("federated_xgboost", "Federated XGBoost (deployed model)",
         "No patient record ever leaves either hospital. Each hospital "
         "trains its own XGBoost model locally; only the trained models' "
         "output margins (a handful of numbers, not patient data) are "
         "averaged centrally. This is the model actually used for scoring "
         "in this app."),
        ("fedavg_logistic", "FedAvg (logistic regression)",
         "A second, independent federated approach: a simple logistic "
         "model whose weight vector is averaged across hospitals over "
         "several training rounds (the textbook FedAvg algorithm, McMahan "
         "et al. 2017). Included to show the federated averaging math "
         "explicitly, since XGBoost trees have no such weight vector."),
    ]
    rows = []
    for key, label, why in rows_spec:
        m = _REGIME_METRICS.get(key)
        if not m:
            continue
        auc = f"{m['roc_auc']*100:.1f}%"
        pr = f"{m['pr_auc']*100:.1f}%"
        rows.append(f"""
        <tr>
          <td style="padding:8px;font-weight:600;color:#0b0b0b">{label}</td>
          <td style="padding:8px;text-align:right;color:#0b0b0b">{auc}</td>
          <td style="padding:8px;text-align:right;color:#52514e">{pr}</td>
          <td style="padding:8px;color:#52514e;font-size:12.5px">{why}</td>
        </tr>""")
    return (
        "<table style='width:100%;border-collapse:collapse;font-size:13px'>"
        "<tr style='color:#52514e;border-bottom:1px solid #e3e1dc'>"
        "<th align=left style='padding:8px'>Training regime</th>"
        "<th align=right style='padding:8px'>ROC-AUC</th>"
        "<th align=right style='padding:8px'>PR-AUC</th>"
        "<th align=left style='padding:8px'>What it means</th></tr>"
        + "".join(rows) + "</table>"
    )


_SCORE_EXPLAINER = """
#### What "Risk %" means

Each patient's **Risk %** is the model's predicted probability that they'll
be readmitted within 30 days of this discharge, calibrated so it lines up
with reality — e.g. if 100 patients are shown at ~20% risk, roughly 20 of
them are expected to actually come back within 30 days. It is **not** a
guess dressed up as a percentage: the model (XGBoost) is trained on prior
discharges where the 30-day outcome is already known, and it outputs the
probability of the same pattern repeating.

**Risk bands** (used for the color coding and sorting throughout this app):

| Band | Cutoff | Meaning |
|---|---|---|
| <span style="color:#0ca30c">■</span> **LOW** | < 15% | Routine discharge planning |
| <span style="color:#e8c209">■</span> **MODERATE** | 15–35% | Worth a second look at discharge |
| <span style="color:#e87f0e">■</span> **ELEVATED** | 35–60% | Care-team review recommended |
| <span style="color:#d03b3b">■</span> **HIGH** | ≥ 60% | Strong candidate for a discharge-risk intervention |

**ROC-AUC** measures how well the model *ranks* patients from
lowest-to-highest risk (1.0 = perfect ranking, 0.5 = a coin flip) — it
answers "if you picked one patient who was readmitted and one who wasn't,
how often would the model score the readmitted one higher?" **PR-AUC**
(precision-recall AUC) is a stricter measure that matters more here because
readmission is rare (~10% of discharges): it captures how well the model
finds the true positives without drowning in false alarms.

#### Why local / centralized / federated?

Two hospitals in this simulation see different patient populations
(Hospital A skews older and cardiac-heavy; Hospital B skews younger and
metabolic-heavy). HIPAA prevents them from pooling raw patient records to
train one shared model. The table below compares what's achievable at
each hospital alone (**local**) against the illegal-but-informative upper
bound of pooling all the data (**centralized**), and shows that
**federated learning** — sharing only trained model parameters, never
patient rows — closes almost the entire gap.
"""

overview_pane = pn.Column(
    pn.pane.Markdown("## Cohort overview — held-out test set"),
    pn.Row(
        _stat_card(f"{N_PATIENTS:,}", "Patients in held-out set"),
        _stat_card(f"{ACTUAL_READMIT_RATE:.1%}", "Actual 30-day readmit rate"),
        _stat_card(f"{MEAN_RISK:.1%}", "Mean predicted risk"),
        _stat_card(f"{PCT_HIGH_OR_ELEVATED:.1%}", "Flagged ELEVATED or HIGH",
                   color=BAND_COLORS["HIGH"]),
    ),
    pn.pane.Markdown("#### Risk band distribution"),
    pn.pane.HTML(_band_bar_html()),
    pn.pane.Markdown(_SCORE_EXPLAINER),
    pn.pane.Markdown("#### Training regimes — what each one means"),
    pn.pane.HTML(_regime_table_html()),
    pn.pane.Markdown("#### Regime comparison chart (held-out set)"),
    (pn.pane.PNG(str(REPORTS_DIR / "regime_comparison.png"),
                sizing_mode="scale_width")
     if (REPORTS_DIR / "regime_comparison.png").exists()
     else pn.pane.Markdown("_Run scripts/train.py to produce the regime chart._")),
)

# ── Patient roster (Tabulator) ──────────────────────────────────────────────

roster_table = pn.widgets.Tabulator(
    ROSTER_VIEW,
    pagination="local",
    page_size=25,
    selectable=False,
    disabled=True,
    show_index=False,
    hidden_columns=["row_id"],
    sizing_mode="stretch_width",
    formatters={"Risk %": {"type": "progress", "max": 100, "legend": True,
                           "color": ["#0ca30c", "#e8c209", "#e87f0e", "#d03b3b"]}},
    widths={"Patient": 200, "Site": 70, "Risk %": 140, "Band": 100,
            "Age": 70, "Sex": 60, "Prior adm.": 100, "Diagnoses": 100,
            "LOS (d)": 90},
)


def _style_band_col(row):
    color = BAND_COLORS.get(row["Band"], "#0b0b0b")
    return [f"color: {color}; font-weight: 700" if col == "Band" else ""
            for col in row.index]


roster_table.style.apply(_style_band_col, axis=1)

roster_search = pn.widgets.TextInput(
    name="Filter by patient ID or site", placeholder="e.g. A- or B-")
roster_band_filter = pn.widgets.MultiChoice(
    name="Risk band", options=["LOW", "MODERATE", "ELEVATED", "HIGH"],
    value=[])


@pn.depends(_search_value=roster_search.param.value)
def _search_filter(df, _search_value=None):
    # add_filter always invokes this as filt(df) — Panel's dependency
    # injection does not apply here, so _search_value is always None at
    # call time. @pn.depends(kw=...) only serves to register a *keyword*
    # dependency (Tabulator.add_filter reads filter._dinfo['kw']) so
    # add_filter knows to re-run filtering when roster_search.value
    # changes; the live value must still be read from the widget itself.
    needle = roster_search.value.strip().lower()
    if not needle:
        return df
    return df[df["Patient"].str.lower().str.contains(needle)
              | df["Site"].str.lower().str.contains(needle)]


# Tabulator's own add_filter (not reassigning .value) keeps its internal
# row index-mapping intact under filtering — reassigning .value directly
# was desyncing on_click's row resolution, showing the wrong patient.
roster_table.add_filter(_search_filter)
roster_table.add_filter(roster_band_filter, column="Band")

detail_placeholder = pn.pane.Markdown(
    "_Click a row above to view that patient's risk breakdown, SHAP "
    "drivers, clinical note, and deliberation transcript._")
detail_container = pn.Column(detail_placeholder)

patients_page = pn.Column(
    pn.pane.Markdown("## Patients — held-out test set"),
    pn.Row(roster_search, roster_band_filter),
    roster_table,
    pn.layout.Divider(),
    pn.pane.Markdown("### Patient detail"),
    detail_container,
)

# ── Detail rendering (shared by roster click + bundle-input page) ──────────


def _risk_html(risk: float) -> str:
    band = RISK_BANDS.label(risk)
    color = BAND_COLORS[band]
    pct = f"{risk:.0%}"
    return f"""
    <div style="text-align:center;padding:12px">
      <div style="font-size:42px;font-weight:700;color:{color}">{pct}</div>
      <div style="font-size:14px;color:#52514e">30-day readmission risk —
        <b style="color:{color}">{band}</b></div>
      <div style="background:#eceae6;border-radius:6px;height:10px;margin-top:10px">
        <div style="background:{color};width:{min(risk*100,100):.0f}%;
                    height:10px;border-radius:6px"></div>
      </div>
    </div>"""


def _shap_html(contribs) -> str:
    max_abs = max(abs(c.shap) for c in contribs) or 1.0
    rows = []
    for c in contribs:
        width = abs(c.shap) / max_abs * 46
        color = "#d03b3b" if c.shap > 0 else "#2a78d6"
        side = ("margin-left:50%" if c.shap > 0
                else f"margin-left:{50 - width:.0f}%")
        value = ("yes" if c.value == 1 else "no") if c.unit == "0/1" \
            else f"{c.value:g} {c.unit}"
        rows.append(f"""
        <tr>
          <td style="padding:3px 8px;color:#0b0b0b">{c.label}</td>
          <td style="padding:3px 8px;color:#52514e">{value}</td>
          <td style="width:45%;padding:3px 0">
            <div style="{side};width:{width:.0f}%;height:12px;
                        background:{color};border-radius:3px"></div>
          </td>
          <td style="padding:3px 8px;color:#52514e;text-align:right">
            {c.shap:+.3f}</td>
        </tr>""")
    return (
        "<table style='width:100%;border-collapse:collapse;font-size:13px'>"
        "<tr style='color:#52514e'><th align=left style='padding:3px 8px'>Driver"
        "</th><th align=left style='padding:3px 8px'>Value</th>"
        "<th>← lowers | raises →</th><th align=right style='padding:3px 8px'>"
        "SHAP</th></tr>" + "".join(rows) + "</table>"
    )


def _transcript_html(transcript) -> str:
    blocks = []
    for turn in transcript:
        who = turn["role"]
        color = {"Clinician": "#2a78d6", "RiskAnalyst": "#4a3aa7"}.get(who, "#52514e")
        blocks.append(
            f"<div style='margin:6px 0;padding:8px 10px;border-left:3px solid "
            f"{color};background:#f9f9f7'><b style='color:{color}'>{who}</b><br>"
            f"<span style='font-size:13px;color:#0b0b0b'>{turn['content']}</span></div>"
        )
    return "".join(blocks)


class _OfflineClient:
    """Forces the deterministic template path without probing LM Studio."""
    def complete(self, *a, **k):
        return None


def _build_detail(row: dict, run_deliberation: bool) -> pn.Column:
    contribs = EXPLAINER.explain_row(row)
    risk = row.get("_risk", MODEL.predict_row(row))

    risk_pane = pn.pane.HTML(_risk_html(risk))
    shap_pane = pn.pane.HTML(_shap_html(contribs))

    # Always attempt a real LLM explanation of *why* the score is what it
    # is; this silently falls back to a clearly-labeled deterministic
    # template if LM Studio is unreachable, so the "why" is never empty.
    note = generate_clinical_note(risk, contribs)

    if run_deliberation:
        result = deliberate(risk, contribs)
        transcript_html = _transcript_html(
            [t for t in result["transcript"] if t["role"] != "case"])
    else:
        transcript_html = ("<i style='color:#52514e'>Enable the "
                          "deliberation checkbox to run the two-agent "
                          "Clinician ⇄ Risk Analyst debate.</i>")

    note_pane = pn.pane.Markdown(
        f"**Why this score — clinical rationale** _({note['source']})_\n\n"
        f"{note['note']}")
    transcript_pane = pn.pane.HTML(transcript_html)

    return pn.Column(
        pn.Row(pn.Column(risk_pane, width=340), pn.Column(shap_pane)),
        pn.Card(note_pane, title="Clinical rationale", collapsed=False),
        pn.Card(transcript_pane, title="Clinician ⇄ Risk Analyst deliberation",
                collapsed=True),
    )


roster_run_deliberation = pn.widgets.Checkbox(
    name="Also run Clinician ⇄ Risk Analyst deliberation for selected "
         "patient (slower, needs LM Studio)", value=False)


_selected_patient = {"row_id": None}


_loading_pane = pn.pane.Markdown(
    "_Loading risk breakdown and clinical rationale…_")


def _render_selected_patient():
    row_id = _selected_patient["row_id"]
    if row_id is None:
        detail_container[:] = [detail_placeholder]
        return
    # row_id (not patient_id) is the lookup key: patient_id is not unique
    # in the held-out set (122 rows share a patient_id with another
    # admission of the same patient), so a patient_id lookup can silently
    # resolve to the wrong admission.
    record = TEST_DF.iloc[row_id]
    pid = record["patient_id"]
    row = record[FEATURE_COLUMNS].to_dict()
    row["_risk"] = float(_risk_scores[row_id])
    detail_container[:] = [
        pn.pane.Markdown(f"**{pid}** — Hospital {record['hospital']}"),
        _build_detail(row, roster_run_deliberation.value),
    ]


def _on_roster_row_click(event):
    # Tabulator's cell-click dispatch (panel/widgets/tables.py
    # _process_event) runs this callback synchronously via
    # state.execute(..., schedule=False) rather than scheduling it as a
    # document next-tick callback. The real LLM call inside
    # _render_selected_patient can take 15-20s (LM Studio round trip),
    # and with nothing shown in the meantime it looked exactly like
    # clicking did nothing. pn.io.unlocked() flushes the "loading" state
    # to the browser immediately, before the slow call blocks this
    # callback; pn.io.hold() then batches the final detail update.
    row_id = int(roster_table.value.iloc[event.row]["row_id"])
    _selected_patient["row_id"] = row_id
    with pn.io.unlocked():
        detail_container[:] = [_loading_pane]
    with pn.io.hold():
        _render_selected_patient()


roster_table.on_click(_on_roster_row_click, column="Patient")
roster_run_deliberation.param.watch(lambda e: _render_selected_patient(), "value")
patients_page.insert(2, roster_run_deliberation)

# ── Bundle input page (ad-hoc FHIR scoring) ─────────────────────────────────

bundle_input = pn.widgets.TextAreaInput(
    name="Paste a FHIR R4 Bundle (JSON)", height=200,
    placeholder='{"resourceType": "Bundle", ...}',
)
bundle_run_deliberation = pn.widgets.Checkbox(
    name="Also run Clinician ⇄ Risk Analyst deliberation (slower, needs "
         "LM Studio)", value=False)
bundle_analyze_btn = pn.widgets.Button(name="Assess readmission risk",
                                       button_type="primary")
bundle_status = pn.pane.Markdown("")
bundle_result = pn.Column()


def _analyze_bundle(_=None):
    bundle_status.object = ""
    try:
        bundle = json.loads(bundle_input.value)
        row = parse_bundle(bundle)
    except (json.JSONDecodeError, BundleParseError) as e:
        bundle_status.object = f"**Input error:** {e}"
        bundle_result[:] = []
        return
    bundle_result[:] = [_build_detail(row, bundle_run_deliberation.value)]


bundle_analyze_btn.on_click(_analyze_bundle)

bundle_page = pn.Column(
    pn.pane.Markdown("## Ad-hoc FHIR Bundle scoring"),
    pn.pane.Markdown(
        "Paste a FHIR R4 `Bundle` (type `collection`) to score a patient "
        "outside the held-out roster — see the README's FHIR Resource "
        "Catalog for the exact resources/fields consumed."),
    bundle_input,
    pn.Row(bundle_run_deliberation, bundle_analyze_btn),
    bundle_status,
    bundle_result,
)

# ── App shell / navigation ───────────────────────────────────────────────────

main_area = pn.Column(overview_pane)

nav = pn.widgets.RadioButtonGroup(
    name="Navigate", options=["Overview", "Patients", "Score a FHIR Bundle"],
    button_type="primary", orientation="vertical",
    sizing_mode="stretch_width",
)


def _on_nav(event):
    main_area[:] = {
        "Overview": [overview_pane],
        "Patients": [patients_page],
        "Score a FHIR Bundle": [bundle_page],
    }[event.new]


nav.param.watch(_on_nav, "value")
nav.value = "Overview"

template = pn.template.FastListTemplate(
    title="ReadmitRadar — 30-day readmission risk at discharge",
    accent_base_color=ACCENT, header_background=ACCENT,
    sidebar=[nav, pn.layout.Divider(),
             pn.pane.Markdown(
                 "**On-prem by design.** Model, SHAP, LLM (LM Studio) "
                 "and tracing (Langfuse) all run locally — no PHI leaves "
                 "the environment.")],
    main=[main_area],
)
template.servable()
