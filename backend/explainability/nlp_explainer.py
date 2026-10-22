"""
Template-based NLP explanation generator for CodeGuard.

Generates plain-English one-sentence explanations from SHAP factor dicts.
Always works — no model download required, no external API.
"""
import random
from typing import Optional


# Risk-tier adjective bundles for varied output
_TIER_PHRASES = {
    "High": [
        "high-risk and should be reviewed urgently",
        "flagged as high-risk — immediate attention is recommended",
        "classified as high defect risk",
        "a high-priority target for code review",
    ],
    "Medium": [
        "moderate-risk and warrants attention",
        "showing moderate defect risk",
        "a medium-risk file worth inspecting",
        "at moderate risk — consider a targeted review",
    ],
    "Low": [
        "low-risk and looks healthy",
        "in good shape with low defect risk",
        "classified as low risk",
        "low-priority from a defect-risk standpoint",
    ],
}

_CONNECTORS = ["mainly because of", "driven primarily by", "largely due to", "chiefly because of"]
_ALSO = ["and also", "compounded by", "along with", "as well as"]


def _pct_label(pct: float) -> str:
    """Convert a SHAP percentage to a descriptive label."""
    if pct >= 40:
        return "strongly"
    elif pct >= 20:
        return "notably"
    elif pct >= 10:
        return "moderately"
    return "slightly"


def generate_explanation(
    risk_tier: str,
    defect_probability: float,
    top_factors: list[dict],
    file_path: Optional[str] = None,
) -> str:
    """
    Generate a one-sentence plain-English explanation.

    Parameters
    ----------
    risk_tier : 'Low', 'Medium', or 'High'
    defect_probability : float in [0, 1]
    top_factors : list of {feature, friendly_name, shap_value, pct}
    file_path : optional file name for personalisation
    """
    tier_phrase = random.choice(_TIER_PHRASES.get(risk_tier, _TIER_PHRASES["Low"]))
    prob_pct = round(defect_probability * 100, 1)

    subject = f"**{file_path}**" if file_path else "This module"

    if not top_factors:
        return (
            f"{subject} is {tier_phrase} (defect probability: {prob_pct}%). "
            "No feature attribution data is available."
        )

    connector = random.choice(_CONNECTORS)
    parts = []
    for i, factor in enumerate(top_factors[:3]):
        fn = factor.get("friendly_name", factor.get("feature", "unknown metric"))
        pct = factor.get("pct", 0.0)
        sv  = factor.get("shap_value", 0.0)
        direction = "high" if sv >= 0 else "low"
        strength  = _pct_label(pct)
        parts.append(f"{direction} **{fn}** ({pct:.0f}%)")

    if len(parts) == 1:
        factor_str = connector + " " + parts[0]
    elif len(parts) == 2:
        also = random.choice(_ALSO)
        factor_str = f"{connector} {parts[0]} {also} {parts[1]}"
    else:
        also = random.choice(_ALSO)
        factor_str = f"{connector} {parts[0]}, {parts[1]}, {also} {parts[2]}"

    return (
        f"{subject} is {tier_phrase} ({prob_pct}% defect probability), "
        f"{factor_str}."
    )


def generate_recommendation(risk_tier: str, top_factors: list[dict]) -> str:
    """
    Generate a short, actionable recommendation based on the top driver.
    """
    recs = {
        "High": {
            "default": (
                "Prioritise this file for code review. Consider refactoring to reduce "
                "complexity and coupling before merging."
            ),
            "loc":    "Break this file into smaller modules to reduce size and coupling.",
            "wmc":    "Decompose high-complexity methods using the Extract Method pattern.",
            "cbo":    "Reduce tight coupling by introducing interfaces or dependency injection.",
            "max_cc": "Simplify conditional logic; consider splitting methods with high cyclomatic complexity.",
            "avg_cc": "Reduce average method complexity — aim for CC ≤ 10 per method.",
            "lcom":   "Improve cohesion: methods that do not share data may belong in separate classes.",
            "rfc":    "Reduce the number of methods called from this class to lower coupling.",
            "cyclomatic_complexity": "Simplify branching logic; target cyclomatic complexity below 15.",
            "coupling":              "Decouple from dependent files using interfaces or dependency injection.",
            "churn_count":           "High churn indicates instability — add regression tests for this file.",
        },
        "Medium": {
            "default": (
                "Consider adding unit tests for this file and scheduling a targeted review "
                "in the next sprint."
            ),
            "loc":    "Monitor size growth; consider splitting if functionality expands.",
            "wmc":   "Keep an eye on method complexity — test edge cases thoroughly.",
            "cbo":   "Document the coupling dependencies to make future refactoring easier.",
        },
        "Low": {
            "default": "No immediate action required. Continue standard review practices.",
        },
    }

    tier_recs = recs.get(risk_tier, recs["Low"])
    if top_factors:
        top_feature = top_factors[0].get("feature", "")
        rec = tier_recs.get(top_feature, tier_recs.get("default", tier_recs["default"]))
    else:
        rec = tier_recs.get("default", "No recommendation available.")

    return rec
