"""SHAP explainer and offline note/deliberation fallbacks."""
import pytest

from llm.clinical_note import _template_note
from llm.deliberation import _template_deliberation
from ml.explain import RiskExplainer, contributions_to_text
from ml.model import ReadmissionModel


@pytest.fixture(scope="module")
def fitted_model(hospital_a):
    return ReadmissionModel({"n_estimators": 60, "max_depth": 3}).fit(hospital_a)


@pytest.fixture(scope="module")
def contributions(fitted_model, test_set):
    explainer = RiskExplainer(fitted_model)
    return explainer.explain_row(test_set.iloc[0].to_dict(), top_k=6)


def test_top_k_contributions(contributions):
    assert len(contributions) == 6
    # Sorted by |shap| descending
    magnitudes = [abs(c.shap) for c in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_contribution_directions(contributions):
    for c in contributions:
        assert c.direction == ("raises" if c.shap > 0 else "lowers")


def test_contributions_text_render(contributions):
    text = contributions_to_text(contributions)
    assert text.count("\n") == 5
    assert "SHAP" in text


def test_template_note_marks_fallback(contributions):
    note = _template_note(0.72, "HIGH", contributions)
    assert "HIGH risk" in note
    assert "fallback" in note.lower()


def test_template_deliberation_structure(contributions):
    result = _template_deliberation(0.72, contributions)
    assert result["source"] == "template"
    roles = [t["role"] for t in result["transcript"]]
    assert roles == ["case", "Clinician", "RiskAnalyst"]


def test_predict_row_matches_batch(fitted_model, test_set):
    row = test_set.iloc[0].to_dict()
    single = fitted_model.predict_row(row)
    batch = fitted_model.predict_proba(test_set.head(1))[0]
    assert single == pytest.approx(float(batch))
