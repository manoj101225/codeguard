"""
CodeGuard seed script.

Downloads PROMISE datasets, registers projects in the DB,
trains the TCA+ model on camel-1.6 → jedit-4.0,
and generates predictions — so the dashboard has live data immediately.

Run: python seed.py
"""
import sys
import os
import json
import logging

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    from backend.ml.data_loader import download_all, load_dataset
    from backend.db.models import get_engine, get_session, Project, ModelRun
    from backend.ml.train import train, multi_source_comparison
    from backend.ml.data_loader import get_features_labels, FEATURE_COLS
    from backend.explainability.shap_engine import (
        SHAPEngine, compute_historical_correlations, strongest_historical_predictor
    )
    from backend.explainability.nlp_explainer import generate_explanation, generate_recommendation
    from backend.reports.debt_estimator import (
        estimate_tech_debt, project_health_score, estimated_post_release_risk
    )
    from backend.ml.train import assign_risk_tier, load_artefact
    from backend.db.models import FileMetrics, Prediction

    # ── 1. Initialize DB ──────────────────────────────────────────────
    log.info("Initialising database …")
    get_engine()

    # ── 2. Download all PROMISE datasets ─────────────────────────────
    log.info("Downloading PROMISE datasets …")
    results = download_all()
    for name, r in results.items():
        if r["ok"]:
            log.info("  ✓ %s (%d rows)", name, r["rows"])
        else:
            log.warning("  ✗ %s — %s", name, r.get("error"))

    # ── 3. Register projects ──────────────────────────────────────────
    session = get_session()

    def get_or_create_project(name, role):
        p = session.query(Project).filter_by(name=name).first()
        if not p:
            p = Project(name=name, role=role)
            session.add(p)
            session.commit()
            log.info("  Registered project: %s (%s) id=%d", name, role, p.id)
        return p.id  # return plain int to avoid DetachedInstanceError

    src_id   = get_or_create_project("camel-1.6",  "source")
    tgt_id   = get_or_create_project("jedit-4.0",  "target")
    tgt2_id  = get_or_create_project("ivy-2.0",    "target")
    msrc1_id = get_or_create_project("lucene-2.4", "source")
    msrc2_id = get_or_create_project("poi-2.5",    "source")
    session.close()

    # ── 4. Train: with transfer learning ─────────────────────────────
    log.info("Training TCA+ model: camel-1.6 → jedit-4.0 …")
    result_tl = train(
        source_project="camel-1.6",
        target_project="jedit-4.0",
        use_transfer=True,
        model_type="rf",
        n_splits=10,
        persist=True,
        db_source_id=src_id,
        db_target_id=tgt_id,
    )
    log.info(
        "  TCA+ F1=%.3f  precision=%.3f  recall=%.3f  p=%.4f",
        result_tl["f1_score"], result_tl["precision_score"],
        result_tl["recall_score"], result_tl["p_value"]
    )

    # ── 5. Train: baseline (no transfer) for comparison ───────────────
    log.info("Training baseline (no transfer) for comparison …")
    result_base = train(
        source_project="camel-1.6",
        target_project="jedit-4.0",
        use_transfer=False,
        model_type="rf",
        n_splits=10,
        persist=True,
        db_source_id=src_id,
        db_target_id=tgt_id,
    )
    log.info(
        "  Baseline F1=%.3f  Improvement=+%.3f",
        result_base["f1_score"],
        result_tl["f1_score"] - result_base["f1_score"],
    )

    # ── 6. Generate predictions for jedit-4.0 ─────────────────────────
    log.info("Generating predictions for jedit-4.0 …")
    model_path = result_tl.get("model_path")
    if not model_path or not os.path.exists(model_path):
        log.error("Model path not found — skipping prediction generation.")
        return

    artefact = load_artefact(model_path)
    clf = artefact["clf"]
    feat_names = artefact["feature_names"]
    med_thresh = artefact["medium_threshold"]
    high_thresh = artefact["high_threshold"]

    src_df = load_dataset("camel-1.6")
    tgt_df = load_dataset("jedit-4.0")
    X_src, y_src, src_feats = get_features_labels(src_df)
    X_tgt, y_tgt, tgt_feats = get_features_labels(tgt_df)

    common = [c for c in FEATURE_COLS if c in feat_names and c in tgt_feats]
    src_idx = [feat_names.index(c) for c in common if c in feat_names]
    tgt_idx = [tgt_feats.index(c) for c in common]

    X_tgt_c = X_tgt[:, tgt_idx]
    X_src_c = X_src[:, [src_feats.index(c) for c in common if c in src_feats]]

    tca = artefact["tca"]
    _, Z_tgt = tca.fit_transform(X_src_c, X_tgt_c)
    proba = clf.predict_proba(Z_tgt)[:, 1]

    shap_eng = SHAPEngine(clf, common)
    try:
        factors_list = shap_eng.explain_batch(Z_tgt, Z_tgt[:min(50, len(Z_tgt))])
    except Exception:
        factors_list = [[] for _ in range(len(Z_tgt))]

    hist_corr = compute_historical_correlations(X_src_c, y_src, common)
    predictor_sentence = strongest_historical_predictor(hist_corr)
    log.info("  %s", predictor_sentence.replace("**", ""))

    name_col = "name" if "name" in tgt_df.columns else (
        "class" if "class" in tgt_df.columns else None
    )

    session = get_session()
    run = session.query(ModelRun).filter_by(
        source_project_id=src_id, target_project_id=tgt_id,
        used_transfer_learning=True
    ).order_by(ModelRun.trained_at.desc()).first()

    predictions_list = []
    for i, prob in enumerate(proba):
        tier = assign_risk_tier(float(prob), med_thresh, high_thresh)
        factors = factors_list[i] if i < len(factors_list) else []
        file_name = str(tgt_df.iloc[i][name_col]) if name_col else f"module_{i}"
        metrics_dict = {c: float(X_tgt_c[i, j]) for j, c in enumerate(common)}

        explanation  = generate_explanation(tier, float(prob), factors, file_name)
        recommendation = generate_recommendation(tier, factors)

        loc = int(metrics_dict.get("loc", 0))
        cc  = int(metrics_dict.get("max_cc", metrics_dict.get("avg_cc", 0)))
        cpl = int(metrics_dict.get("cbo", metrics_dict.get("ce", 0)))
        debt = estimate_tech_debt(cc, loc, cpl, tier)

        fm = FileMetrics(
            target_project_id=tgt_id,
            file_path=file_name,
            loc=loc,
            cyclomatic_complexity=cc,
            coupling=cpl,
            num_methods=int(metrics_dict.get("npm", 0)),
            churn_count=0,
        )
        session.add(fm)
        session.flush()

        pred = Prediction(
            model_run_id=run.id if run else None,
            file_metrics_id=fm.id,
            defect_probability=float(prob),
            risk_tier=tier,
            nlp_explanation=explanation,
            recommendation=recommendation,
            tech_debt_hours=debt,
        )
        pred.set_top_factors(factors[:3])
        session.add(pred)

        predictions_list.append({
            "file_path": file_name,
            "defect_probability": round(float(prob), 4),
            "risk_tier": tier,
            "tech_debt_hours": debt,
        })

    session.commit()
    session.close()

    health = project_health_score(predictions_list)
    post_risk = estimated_post_release_risk(predictions_list)
    high_c = sum(1 for p in predictions_list if p["risk_tier"] == "High")
    med_c  = sum(1 for p in predictions_list if p["risk_tier"] == "Medium")
    low_c  = sum(1 for p in predictions_list if p["risk_tier"] == "Low")

    log.info("  Predictions: %d total  🔴 %d High  🟠 %d Medium  🟢 %d Low",
             len(predictions_list), high_c, med_c, low_c)
    log.info("  Project Health Score: %d/100", health)
    log.info("  Post-Release Risk: %.2f", post_risk)
    log.info("  F1 improvement (TL vs baseline): +%.3f",
             result_tl["f1_score"] - result_base["f1_score"])
    log.info("  Statistical significance: p=%.4f (%s)",
             result_tl["p_value"], result_tl.get("significance_test", ""))

    print("\n" + "="*60)
    print("[SUCCESS] CodeGuard seed complete!")
    print(f"   Health score  : {health}/100")
    print(f"   F1 (TCA+)     : {result_tl['f1_score']:.3f}")
    print(f"   F1 (baseline) : {result_base['f1_score']:.3f}")
    print(f"   Improvement   : +{result_tl['f1_score'] - result_base['f1_score']:.3f}")
    print(f"   p-value       : {result_tl['p_value']:.4f}")
    print("="*60)
    print("\nNext steps:")
    print("  1. Start the API  :  uvicorn backend.main:app --reload")
    print("  2. Start dashboard:  streamlit run dashboard/app.py")
    print("="*60)


if __name__ == "__main__":
    main()
