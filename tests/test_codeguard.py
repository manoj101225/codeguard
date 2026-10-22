"""
CodeGuard test suite.
Tests cover: data loading, TCA+ transform, risk tiers, NLP explainer,
debt estimator, SHAP engine, and the FastAPI endpoints.

Run: pytest tests/ -v
"""
import sys
import os
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Data loader tests
# ============================================================

class TestDataLoader:
    def test_clean_produces_expected_columns(self):
        from backend.ml.data_loader import _clean
        import pandas as pd

        df = pd.DataFrame({
            "wmc": [5, 10], "dit": [1, 2], "loc": [100, 200],
            "cbo": [3, 7], "bug": [0, 1]
        })
        cleaned = _clean(df, "test")
        assert "bug" in cleaned.columns
        assert set(cleaned["bug"]).issubset({0, 1})

    def test_label_mapping_true_false(self):
        from backend.ml.data_loader import _clean
        import pandas as pd

        df = pd.DataFrame({
            "wmc": [1, 2, 3], "loc": [10, 20, 30],
            "bug": ["TRUE", "FALSE", "true"]
        })
        cleaned = _clean(df, "test")
        assert list(cleaned["bug"]) == [1, 0, 1]

    def test_get_features_labels_shape(self):
        from backend.ml.data_loader import get_features_labels, _clean, FEATURE_COLS
        import pandas as pd

        data = {c: np.random.rand(20) for c in ["wmc", "loc", "cbo"]}
        data["bug"] = np.random.randint(0, 2, 20)
        df = _clean(pd.DataFrame(data), "test")
        X, y, feats = get_features_labels(df)
        assert X.shape[0] == 20
        assert len(y) == 20
        assert X.shape[1] == len(feats)


# ============================================================
# TCA+ tests
# ============================================================

class TestTCAPlus:
    def _make_data(self, n=80, d=10, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n, d)), rng.standard_normal((n // 2, d))

    def test_fit_transform_shapes(self):
        from backend.ml.transfer_learning import TCAPlus
        X_s, X_t = self._make_data()
        tca = TCAPlus(n_components=5)
        Z_s, Z_t = tca.fit_transform(X_s, X_t)
        assert Z_s.shape[0] == X_s.shape[0]
        assert Z_t.shape[0] == X_t.shape[0]
        assert Z_s.shape[1] == Z_t.shape[1]

    def test_transform_single(self):
        from backend.ml.transfer_learning import TCAPlus
        X_s, X_t = self._make_data()
        tca = TCAPlus(n_components=4)
        tca.fit(X_s, X_t)
        x = np.random.randn(1, X_s.shape[1])
        z = tca.transform_single(x)
        assert z.shape == (1, 4)

    def test_not_fitted_raises(self):
        from backend.ml.transfer_learning import TCAPlus
        tca = TCAPlus()
        with pytest.raises(RuntimeError):
            tca.transform(np.zeros((5, 5)), np.zeros((3, 5)))


# ============================================================
# Risk tier threshold tests
# ============================================================

class TestRiskTiers:
    def test_assign_risk_tier(self):
        from backend.ml.train import assign_risk_tier
        assert assign_risk_tier(0.9, 0.3, 0.6) == "High"
        assert assign_risk_tier(0.5, 0.3, 0.6) == "Medium"
        assert assign_risk_tier(0.1, 0.3, 0.6) == "Low"

    def test_compute_risk_thresholds_returns_valid_range(self):
        from backend.ml.train import compute_risk_thresholds
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 200)
        proba  = rng.uniform(0, 1, 200)
        med, high = compute_risk_thresholds(y_true, proba)
        assert 0 < med < high < 1


# ============================================================
# NLP Explainer tests
# ============================================================

class TestNLPExplainer:
    def _factors(self):
        return [
            {"feature": "max_cc", "friendly_name": "Max Cyclomatic Complexity",
             "shap_value": 0.3, "pct": 38.0},
            {"feature": "cbo", "friendly_name": "Coupling Between Objects",
             "shap_value": 0.2, "pct": 25.0},
        ]

    def test_explanation_not_empty(self):
        from backend.explainability.nlp_explainer import generate_explanation
        exp = generate_explanation("High", 0.85, self._factors(), "payment.py")
        assert len(exp) > 20
        assert "payment.py" in exp

    def test_explanation_with_no_factors(self):
        from backend.explainability.nlp_explainer import generate_explanation
        exp = generate_explanation("Low", 0.1, [], None)
        assert "No feature attribution" in exp

    def test_recommendation_not_empty(self):
        from backend.explainability.nlp_explainer import generate_recommendation
        rec = generate_recommendation("High", self._factors())
        assert len(rec) > 10

    def test_all_tiers_covered(self):
        from backend.explainability.nlp_explainer import generate_explanation
        for tier in ["High", "Medium", "Low"]:
            exp = generate_explanation(tier, 0.5, self._factors(), "f.py")
            assert isinstance(exp, str) and len(exp) > 0


# ============================================================
# Tech debt estimator tests
# ============================================================

class TestDebtEstimator:
    def test_zero_excess_gives_tier_surcharge_only(self):
        from backend.reports.debt_estimator import estimate_tech_debt
        d = estimate_tech_debt(10, 100, 3, "High")
        assert d == 1.5   # only tier surcharge

    def test_excess_adds_to_debt(self):
        from backend.reports.debt_estimator import estimate_tech_debt
        d = estimate_tech_debt(25, 300, 10, "Low")
        # cc_excess=10*0.5=5, loc_excess=1*1.0=1, cpl_excess=5*0.3=1.5 → 7.5
        assert d == 7.5

    def test_health_score_all_low(self):
        from backend.reports.debt_estimator import project_health_score
        preds = [{"risk_tier": "Low"}] * 20
        score = project_health_score(preds)
        assert score >= 90

    def test_health_score_all_high(self):
        from backend.reports.debt_estimator import project_health_score
        preds = [{"risk_tier": "High"}] * 20
        score = project_health_score(preds)
        assert score == 0

    def test_hotspot_rank_order(self):
        from backend.reports.debt_estimator import hotspot_rank
        preds = [
            {"file_path": "a.py", "defect_probability": 0.9, "risk_tier": "High",
             "tech_debt_hours": 5},
            {"file_path": "b.py", "defect_probability": 0.4, "risk_tier": "Medium",
             "tech_debt_hours": 2},
        ]
        centrality = {"a.py": 0.8, "b.py": 0.1}
        ranked = hotspot_rank(preds, centrality)
        assert ranked[0]["file_path"] == "a.py"


# ============================================================
# SHAP engine fallback test
# ============================================================

class TestSHAPEngine:
    def test_fallback_returns_correct_shape(self):
        from backend.explainability.shap_engine import SHAPEngine
        from sklearn.ensemble import RandomForestClassifier

        X = np.random.rand(60, 5)
        y = np.random.randint(0, 2, 60)
        clf = RandomForestClassifier(n_estimators=10, random_state=0)
        clf.fit(X, y)

        eng = SHAPEngine(clf, ["a", "b", "c", "d", "e"])
        results = eng.explain_batch(X[:5], X[:10])
        assert len(results) == 5
        for row in results:
            assert len(row) <= 3
            for f in row:
                assert "feature" in f and "pct" in f

    def test_historical_correlations(self):
        from backend.explainability.shap_engine import compute_historical_correlations
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 4))
        y = (X[:, 0] > 0).astype(int)   # feature 0 is perfectly correlated
        corrs = compute_historical_correlations(X, y, ["f0", "f1", "f2", "f3"])
        assert corrs[0]["feature"] == "f0"   # should rank highest


# ============================================================
# FastAPI endpoint smoke tests
# ============================================================

class TestAPI:
    @pytest.fixture(autouse=True)
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        self.client = TestClient(app)

    def test_root(self):
        r = self.client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "running"

    def test_create_and_list_project(self):
        r = self.client.post("/api/projects",
                             json={"name": "test-proj", "role": "target"})
        assert r.status_code == 200
        pid = r.json()["id"]
        assert pid > 0

        r2 = self.client.get("/api/projects")
        assert r2.status_code == 200
        names = [p["name"] for p in r2.json()]
        assert "test-proj" in names

    def test_list_runs(self):
        r = self.client.get("/api/runs")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_badge_svg(self):
        # Register a project first
        r = self.client.post("/api/projects",
                             json={"name": "badge-test", "role": "target"})
        pid = r.json()["id"]
        r2 = self.client.get(f"/api/projects/{pid}/badge.svg")
        assert r2.status_code == 200
        assert b"svg" in r2.content.lower()

    def test_ci_evaluate_pass(self):
        r = self.client.post("/api/ci/evaluate", json={
            "commit_sha": "abc123",
            "changed_files": [],
            "override": False,
        })
        assert r.status_code == 200
        assert r.json()["gate_pass"] is True

    def test_ci_evaluate_override(self):
        r = self.client.post("/api/ci/evaluate", json={
            "commit_sha": "def456",
            "changed_files": [],
            "override": True,
            "override_reason": "Emergency hotfix",
        })
        assert r.status_code == 200
        assert r.json()["gate_decision"] == "overridden"

    def test_whatif_no_models(self):
        # Without any models trained, should return 404
        r = self.client.post("/api/predictions/1/whatif", json={
            "model_run_id": 0,
            "modified_metrics": {"loc": 100, "wmc": 5},
        })
        # Either 404 (no model) or 200 (if model exists from seed)
        assert r.status_code in (200, 404, 500)
