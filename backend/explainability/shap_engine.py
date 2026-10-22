"""
SHAP-based feature attribution engine for CodeGuard.

Computes per-file top-3 SHAP contributions and point-biserial
historical correlations for the "historical predictor" sentence.
"""
import logging
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    log.warning("shap not installed — using fallback feature importance from model.")


FRIENDLY_NAMES = {
    "wmc":    "Weighted Methods per Class",
    "dit":    "Depth of Inheritance",
    "noc":    "Number of Children",
    "cbo":    "Coupling Between Objects",
    "rfc":    "Response for a Class",
    "lcom":   "Lack of Cohesion in Methods",
    "ca":     "Afferent Coupling",
    "ce":     "Efferent Coupling",
    "npm":    "Number of Public Methods",
    "loc":    "Lines of Code",
    "dam":    "Data Access Metric",
    "moa":    "Measure of Aggregation",
    "mfa":    "Measure of Functional Abstraction",
    "cam":    "Cohesion Among Methods",
    "ic":     "Inheritance Coupling",
    "cbm":    "Coupling Between Methods",
    "amc":    "Average Method Complexity",
    "max_cc": "Max Cyclomatic Complexity",
    "avg_cc": "Average Cyclomatic Complexity",
    # Real-repo metrics
    "cyclomatic_complexity": "Cyclomatic Complexity",
    "coupling":              "Coupling to Other Files",
    "num_methods":           "Number of Methods",
    "churn_count":           "Code Churn (commit count)",
}


class SHAPEngine:
    """
    Wraps a trained classifier and a TCA-projected dataset to produce
    SHAP feature attributions per prediction.
    """

    def __init__(self, clf, feature_names: list[str]):
        self.clf = clf
        self.feature_names = feature_names
        self._explainer = None

    def _get_explainer(self, X_background: np.ndarray):
        if not HAS_SHAP:
            return None
        if self._explainer is None:
            try:
                self._explainer = shap.TreeExplainer(self.clf)
            except Exception:
                try:
                    summary = shap.sample(X_background, min(50, len(X_background)))
                    self._explainer = shap.KernelExplainer(
                        self.clf.predict_proba, summary
                    )
                except Exception as e:
                    log.warning("SHAP explainer init failed: %s", e)
                    return None
        return self._explainer

    def explain_batch(
        self,
        X: np.ndarray,
        X_background: np.ndarray,
        top_k: int = 3,
    ) -> list[list[dict]]:
        """
        Returns a list of top_k attribution dicts for each row in X.
        Each dict: {feature, friendly_name, shap_value, pct}
        Falls back to model feature_importances_ if SHAP unavailable.
        """
        explainer = self._get_explainer(X_background)

        if explainer is not None and HAS_SHAP:
            try:
                sv = explainer.shap_values(X)
                # TreeExplainer returns list[array] for multi-class; take class-1 slice
                if isinstance(sv, list):
                    sv = sv[1]
                return self._format_shap(sv, top_k)
            except Exception as e:
                log.warning("SHAP computation failed: %s — falling back to feature importance", e)

        # Fallback: use model's feature_importances_ uniformly
        return self._fallback_importance(X, top_k)

    def _format_shap(self, shap_values: np.ndarray, top_k: int) -> list[list[dict]]:
        results = []
        for row in shap_values:
            abs_vals = np.abs(row)
            total = abs_vals.sum() + 1e-10
            top_idx = np.argsort(abs_vals)[::-1][:top_k]
            factors = []
            for i in top_idx:
                fname = self.feature_names[i] if i < len(self.feature_names) else f"feat_{i}"
                pct = float(abs_vals[i] / total * 100)
                factors.append({
                    "feature": fname,
                    "friendly_name": FRIENDLY_NAMES.get(fname, fname),
                    "shap_value": float(row[i]),
                    "pct": round(pct, 1),
                })
            results.append(factors)
        return results

    def _fallback_importance(self, X: np.ndarray, top_k: int) -> list[list[dict]]:
        try:
            importances = self.clf.feature_importances_
        except AttributeError:
            importances = np.ones(len(self.feature_names)) / len(self.feature_names)

        abs_imp = np.abs(importances)
        total = abs_imp.sum() + 1e-10
        top_idx = np.argsort(abs_imp)[::-1][:top_k]
        base_factors = []
        for i in top_idx:
            fname = self.feature_names[i] if i < len(self.feature_names) else f"feat_{i}"
            pct = float(abs_imp[i] / total * 100)
            base_factors.append({
                "feature": fname,
                "friendly_name": FRIENDLY_NAMES.get(fname, fname),
                "shap_value": float(importances[i]),
                "pct": round(pct, 1),
            })

        # Return the same base factors for every row (feature-level, not instance-level)
        return [base_factors for _ in range(len(X))]


def compute_historical_correlations(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> list[dict]:
    """
    Point-biserial correlation of each feature against the binary defect label
    in the source project's labeled history.

    Returns a sorted list: [{feature, friendly_name, correlation, p_value}]
    """
    results = []
    for i, fname in enumerate(feature_names):
        try:
            r, p = stats.pointbiserialr(y, X[:, i])
            results.append({
                "feature": fname,
                "friendly_name": FRIENDLY_NAMES.get(fname, fname),
                "correlation": round(float(r), 4),
                "p_value": round(float(p), 6),
            })
        except Exception:
            pass

    results.sort(key=lambda d: abs(d["correlation"]), reverse=True)
    return results


def strongest_historical_predictor(correlations: list[dict]) -> str:
    """
    Returns a one-sentence summary of the strongest historical predictor.
    """
    if not correlations:
        return "No historical correlation data available."
    top = correlations[0]
    direction = "higher" if top["correlation"] > 0 else "lower"
    return (
        f"Historically, **{top['friendly_name']}** has been the strongest predictor "
        f"of defects in this codebase (r={top['correlation']:.3f}); "
        f"{direction} values are associated with more defects."
    )
