"""
CodeGuard — Training pipeline.

Handles:
  - Baseline model training (Random Forest + XGBoost) on source project
  - TCA+ transfer learning to target project
  - Multi-source comparison study
  - Statistical significance testing (paired t-test / Wilcoxon)
  - Risk-tier threshold computation from precision-recall curve
  - Model persistence and version tracking
"""
import os
import json
import pickle
import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    precision_recall_curve, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from backend.ml.data_loader import (
    load_dataset, get_features_labels, FEATURE_COLS
)
from backend.ml.transfer_learning import TCAPlus
from backend.db.models import get_session, Project, ModelRun

log = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

VERSION_LOG = os.path.join(MODEL_DIR, "version_log.json")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _save_model(obj: Any, filename: str):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def _load_model(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


def _log_version(entry: dict):
    log_data = []
    if os.path.exists(VERSION_LOG):
        with open(VERSION_LOG) as f:
            log_data = json.load(f)
    log_data.append(entry)
    with open(VERSION_LOG, "w") as f:
        json.dump(log_data, f, indent=2)


def compute_risk_thresholds(y_true: np.ndarray, proba: np.ndarray):
    """
    Compute Low/Medium/High thresholds from the precision-recall curve.
    We pick:
      - high_threshold  = probability at which precision ≥ 0.70  (if achievable)
      - medium_threshold = probability at which precision ≥ 0.50
    Falls back to 0.66 / 0.33 if the curve doesn't reach the targets.
    """
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns one more value than thresholds
    precision_arr = precision_arr[:-1]
    recall_arr = recall_arr[:-1]

    high_thresh = 0.66
    medium_thresh = 0.33

    # High: find lowest threshold where precision ≥ 0.70 AND recall ≥ 0.15
    mask_high = (precision_arr >= 0.70) & (recall_arr >= 0.15)
    if mask_high.any():
        high_thresh = float(thresholds[mask_high][0])

    # Medium: find lowest threshold where precision ≥ 0.50
    mask_med = (precision_arr >= 0.50) & (thresholds < high_thresh)
    if mask_med.any():
        medium_thresh = float(thresholds[mask_med][0])

    # Clamp
    medium_thresh = max(0.10, min(medium_thresh, high_thresh - 0.05))
    return medium_thresh, high_thresh


def assign_risk_tier(prob: float, medium_thresh: float, high_thresh: float) -> str:
    if prob >= high_thresh:
        return "High"
    elif prob >= medium_thresh:
        return "Medium"
    return "Low"


def _build_classifier(model_type: str = "rf"):
    if model_type == "xgb" and HAS_XGB:
        return XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, verbosity=0
        )
    # Default: Random Forest
    return RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train(
    source_project: str = "camel-1.6",
    target_project: str = "jedit-4.0",
    use_transfer: bool = True,
    model_type: str = "rf",
    n_splits: int = 10,
    persist: bool = True,
    db_source_id: int | None = None,
    db_target_id: int | None = None,
) -> dict:
    """
    Train a defect prediction model with optional TCA+ transfer learning.

    Returns a dict with metrics and artefact paths.
    """
    log.info("Loading datasets: source=%s  target=%s", source_project, target_project)
    src_df = load_dataset(source_project)
    tgt_df = load_dataset(target_project)

    X_src, y_src, feat_names = get_features_labels(src_df)
    X_tgt, y_tgt, _          = get_features_labels(tgt_df)

    # Align feature sets (intersection)
    src_feat = [c for c in FEATURE_COLS if c in src_df.columns]
    tgt_feat = [c for c in FEATURE_COLS if c in tgt_df.columns]
    common   = [c for c in FEATURE_COLS if c in src_feat and c in tgt_feat]
    src_idx  = [src_feat.index(c) for c in common]
    tgt_idx  = [tgt_feat.index(c) for c in common]
    X_src = X_src[:, src_idx]
    X_tgt = X_tgt[:, tgt_idx]
    feat_names = common

    # ----------------------------------------------------------------
    # Statistical significance: 10 random splits with/without transfer
    # ----------------------------------------------------------------
    f1_with, f1_without = [], []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    tca = TCAPlus(n_components=min(10, len(common) - 1), mu=0.5)

    for fold, (tgt_train_idx, tgt_test_idx) in enumerate(skf.split(X_tgt, y_tgt)):
        X_tgt_tr, y_tgt_tr = X_tgt[tgt_train_idx], y_tgt[tgt_train_idx]
        X_tgt_te, y_tgt_te = X_tgt[tgt_test_idx],  y_tgt[tgt_test_idx]

        # ---- Without transfer: train only on source, test on target ----
        clf_base = _build_classifier(model_type)
        scaler = StandardScaler()
        X_src_sc = scaler.fit_transform(X_src)
        X_tgt_te_sc = scaler.transform(X_tgt_te)
        clf_base.fit(X_src_sc, y_src)
        p_base = clf_base.predict(X_tgt_te_sc)
        f1_without.append(f1_score(y_tgt_te, p_base, zero_division=0))

        # ---- With TCA+ transfer ----
        tca_fold = TCAPlus(n_components=min(10, len(common) - 1), mu=0.5)
        Z_src, Z_tgt_all = tca_fold.fit_transform(X_src, X_tgt)
        Z_tgt_tr = Z_tgt_all[tgt_train_idx]
        Z_tgt_te = Z_tgt_all[tgt_test_idx]

        # Combine projected source + small labeled target (fine-tuning)
        X_train_combined = np.vstack([Z_src, Z_tgt_tr])
        y_train_combined = np.concatenate([y_src, y_tgt_tr])

        clf_tca = _build_classifier(model_type)
        clf_tca.fit(X_train_combined, y_train_combined)
        p_tca = clf_tca.predict(Z_tgt_te)
        f1_with.append(f1_score(y_tgt_te, p_tca, zero_division=0))

    # Significance test
    _, normality_p = stats.shapiro(np.array(f1_with) - np.array(f1_without))
    if normality_p > 0.05:
        stat, p_value = stats.ttest_rel(f1_with, f1_without)
        test_name = "paired t-test"
    else:
        stat, p_value = stats.wilcoxon(f1_with, f1_without)
        test_name = "Wilcoxon signed-rank"

    log.info(
        "%s: stat=%.4f  p=%.4f  F1_with=%.3f  F1_without=%.3f",
        test_name, stat, p_value, np.mean(f1_with), np.mean(f1_without)
    )

    # ----------------------------------------------------------------
    # Final model: train on all source (+ full target if transfer) for deployment
    # ----------------------------------------------------------------
    if use_transfer:
        tca_final = TCAPlus(n_components=min(10, len(common) - 1), mu=0.5)
        Z_src_f, Z_tgt_f = tca_final.fit_transform(X_src, X_tgt)
        X_train_f = np.vstack([Z_src_f, Z_tgt_f])
        y_train_f = np.concatenate([y_src, y_tgt])
    else:
        scaler_f = StandardScaler()
        X_train_f = scaler_f.fit_transform(X_src)
        y_train_f = y_src

    clf_final = _build_classifier(model_type)
    clf_final.fit(X_train_f, y_train_f)

    # Risk thresholds on held-out target
    if use_transfer:
        proba_tgt = clf_final.predict_proba(Z_tgt_f)[:, 1]
    else:
        proba_tgt = clf_final.predict_proba(scaler_f.transform(X_tgt))[:, 1]

    med_thresh, high_thresh = compute_risk_thresholds(y_tgt, proba_tgt)

    # Final metrics (using the cross-val averages as the reported numbers)
    final_f1  = float(np.mean(f1_with if use_transfer else f1_without))
    final_pre = float(precision_score(y_tgt, (proba_tgt >= 0.5).astype(int), zero_division=0))
    final_rec = float(recall_score(y_tgt, (proba_tgt >= 0.5).astype(int), zero_division=0))

    artefact = {
        "clf": clf_final,
        "feature_names": feat_names,
        "medium_threshold": med_thresh,
        "high_threshold": high_thresh,
        "use_transfer": use_transfer,
    }
    if use_transfer:
        artefact["tca"] = tca_final
    else:
        artefact["scaler"] = scaler_f

    result = {
        "source": source_project,
        "target": target_project,
        "model_type": model_type,
        "use_transfer": use_transfer,
        "f1_score": final_f1,
        "precision_score": final_pre,
        "recall_score": final_rec,
        "p_value": float(p_value),
        "significance_test": test_name,
        "f1_with_transfer": float(np.mean(f1_with)),
        "f1_without_transfer": float(np.mean(f1_without)),
        "f1_improvement": float(np.mean(f1_with) - np.mean(f1_without)),
        "medium_threshold": med_thresh,
        "high_threshold": high_thresh,
        "feature_names": feat_names,
        "trained_at": datetime.utcnow().isoformat(),
    }

    if persist:
        tag = f"{source_project}__{target_project}__{model_type}__{'tl' if use_transfer else 'base'}"
        tag = tag.replace(".", "_").replace("-", "_")
        model_path = _save_model(artefact, f"{tag}.pkl")
        result["model_path"] = model_path
        _log_version({**result, "model_path": model_path})
        log.info("Model saved → %s", model_path)

    # Persist to DB
    if db_source_id is not None and db_target_id is not None:
        try:
            session = get_session()
            run = ModelRun(
                source_project_id=db_source_id,
                target_project_id=db_target_id,
                used_transfer_learning=use_transfer,
                f1_score=final_f1,
                precision_score=final_pre,
                recall_score=final_rec,
                p_value=float(p_value),
            )
            session.add(run)
            session.commit()
            result["db_run_id"] = run.id
            session.close()
        except Exception as e:
            log.warning("DB write failed: %s", e)

    return result


# ---------------------------------------------------------------------------
# Multi-source comparison
# ---------------------------------------------------------------------------

def multi_source_comparison(
    sources: list[str] = ("camel-1.6", "lucene-2.4", "poi-2.5"),
    target: str = "jedit-4.0",
    model_type: str = "rf",
) -> list[dict]:
    """Train with each source project and report F1-score."""
    results = []
    for src in sources:
        try:
            r = train(
                source_project=src,
                target_project=target,
                use_transfer=True,
                model_type=model_type,
                n_splits=5,
                persist=False,
            )
            results.append({"source": src, "target": target, "f1": r["f1_score"]})
        except Exception as e:
            log.warning("Multi-source %s → %s failed: %s", src, target, e)
            results.append({"source": src, "target": target, "f1": None, "error": str(e)})
    return results


# ---------------------------------------------------------------------------
# Cross-project generalization study (zero-shot)
# ---------------------------------------------------------------------------

def cross_project_generalization(
    source: str = "camel-1.6",
    targets: list[str] = ("jedit-4.0", "ivy-2.0", "lucene-2.4"),
    model_type: str = "rf",
) -> list[dict]:
    """
    Train ONLY on source, evaluate directly on each target (no fine-tuning).
    Shows where transfer stops helping.
    """
    src_df = load_dataset(source)
    X_src, y_src, src_feats = get_features_labels(src_df)

    results = []
    for tgt in targets:
        try:
            tgt_df = load_dataset(tgt)
            X_tgt, y_tgt, tgt_feats = get_features_labels(tgt_df)

            common = [c for c in FEATURE_COLS if c in src_feats and c in tgt_feats]
            si = [src_feats.index(c) for c in common]
            ti = [tgt_feats.index(c) for c in common]

            scaler = StandardScaler()
            clf = _build_classifier(model_type)
            clf.fit(scaler.fit_transform(X_src[:, si]), y_src)
            p = clf.predict(scaler.transform(X_tgt[:, ti]))
            f1 = f1_score(y_tgt, p, zero_division=0)
            results.append({"source": source, "target": tgt, "f1_zero_shot": f1})
        except Exception as e:
            results.append({"source": source, "target": tgt, "f1_zero_shot": None, "error": str(e)})

    return results


def load_artefact(model_path: str) -> dict:
    """Load a persisted model artefact."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def list_versions() -> list[dict]:
    """Return the version log entries."""
    if not os.path.exists(VERSION_LOG):
        return []
    with open(VERSION_LOG) as f:
        return json.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = train("camel-1.6", "jedit-4.0", use_transfer=True)
    print(json.dumps({k: v for k, v in r.items() if k != "feature_names"}, indent=2))
