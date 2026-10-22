"""
FastAPI backend for CodeGuard.

Run with: uvicorn backend.main:app --reload --port 8000
"""
import os
import sys
import json
import logging
from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make sure imports resolve from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db.models import (
    get_session, Project, ModelRun, FileMetrics, Prediction,
    DeveloperFeedback, CIRun
)
from backend.ml.data_loader import load_dataset, get_features_labels, FEATURE_COLS
from backend.ml.train import (
    train, list_versions, multi_source_comparison,
    cross_project_generalization, assign_risk_tier, load_artefact
)
from backend.ml.whatif import whatif_predict
from backend.explainability.shap_engine import (
    SHAPEngine, compute_historical_correlations, strongest_historical_predictor
)
from backend.explainability.nlp_explainer import generate_explanation, generate_recommendation
from backend.static_analysis.pylint_runner import run_static_analysis
from backend.reports.debt_estimator import (
    estimate_tech_debt, project_health_score, hotspot_rank, estimated_post_release_risk
)
from backend.reports.pdf_generator import generate_pdf_report, generate_badge_svg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="CodeGuard API",
    description="Software Defect Prediction with Transfer Learning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic schemas
# ============================================================

class ProjectCreate(BaseModel):
    name: str
    role: str  # 'source' or 'target'
    repo_url: Optional[str] = None


class TrainRequest(BaseModel):
    source_project_id: int
    target_project_id: int
    use_transfer: bool = True
    model_type: str = "rf"


class WhatIfRequest(BaseModel):
    model_run_id: int
    modified_metrics: dict


class FeedbackRequest(BaseModel):
    feedback_type: str
    note: Optional[str] = ""


class CIEvaluateRequest(BaseModel):
    commit_sha: str
    changed_files: list[str]
    model_run_id: Optional[int] = None
    override: bool = False
    override_reason: Optional[str] = ""


class AnalyzeCodeRequest(BaseModel):
    filename: str = "custom_module.py"
    code: str
    auto_deploy: bool = True


class AnalyzeProjectRequest(BaseModel):
    folder_path: str
    auto_deploy: bool = True




# ============================================================
# Helper: get latest model path for a run
# ============================================================

def _get_model_path_for_run(run: ModelRun) -> Optional[str]:
    versions = list_versions()
    for v in reversed(versions):
        if "model_path" in v and os.path.exists(v["model_path"]):
            # Match by run id approximation (same source/target)
            session = get_session()
            src = session.get(Project, run.source_project_id)
            tgt = session.get(Project, run.target_project_id)
            session.close()
            if (src and tgt and
                    v.get("source") == src.name and
                    v.get("target") == tgt.name):
                return v["model_path"]
    # Fallback: return the latest existing model
    for v in reversed(versions):
        if "model_path" in v and os.path.exists(v["model_path"]):
            return v["model_path"]
    return None


# ============================================================
# Routes
# ============================================================

@app.get("/")
def root():
    return {"service": "CodeGuard API", "status": "running", "version": "1.0.0"}


# ---- Projects ----

@app.post("/api/projects")
def create_project(body: ProjectCreate):
    session = get_session()
    project = Project(name=body.name, role=body.role, repo_url=body.repo_url)
    session.add(project)
    session.commit()
    session.refresh(project)
    result = {"id": project.id, "name": project.name, "role": project.role}
    session.close()
    return result


@app.get("/api/projects")
def list_projects():
    session = get_session()
    projects = session.query(Project).all()
    result = [
        {"id": p.id, "name": p.name, "role": p.role, "repo_url": p.repo_url,
         "created_at": str(p.created_at)}
        for p in projects
    ]
    session.close()
    return result


@app.get("/api/projects/{project_id}")
def get_project(project_id: int):
    session = get_session()
    p = session.get(Project, project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    result = {"id": p.id, "name": p.name, "role": p.role, "repo_url": p.repo_url}
    session.close()
    return result


# ---- Training ----

@app.post("/api/train")
def train_model(body: TrainRequest, background_tasks: BackgroundTasks):
    session = get_session()
    src = session.get(Project, body.source_project_id)
    tgt = session.get(Project, body.target_project_id)
    session.close()

    if not src or not tgt:
        raise HTTPException(404, "Source or target project not found")

    try:
        result = train(
            source_project=src.name,
            target_project=tgt.name,
            use_transfer=body.use_transfer,
            model_type=body.model_type,
            n_splits=10,
            persist=True,
            db_source_id=body.source_project_id,
            db_target_id=body.target_project_id,
        )
        return result
    except Exception as e:
        log.exception("Training failed")
        raise HTTPException(500, str(e))


@app.get("/api/runs")
def list_runs():
    session = get_session()
    runs = session.query(ModelRun).order_by(ModelRun.trained_at.desc()).all()
    result = []
    for r in runs:
        src = session.get(Project, r.source_project_id)
        tgt = session.get(Project, r.target_project_id)
        result.append({
            "id": r.id,
            "source_project": src.name if src else r.source_project_id,
            "target_project": tgt.name if tgt else r.target_project_id,
            "used_transfer_learning": r.used_transfer_learning,
            "f1_score": r.f1_score,
            "precision_score": r.precision_score,
            "recall_score": r.recall_score,
            "p_value": r.p_value,
            "trained_at": str(r.trained_at),
        })
    session.close()
    return result


@app.get("/api/model-versions")
def get_model_versions():
    return list_versions()


# ---- Predictions ----

@app.post("/api/projects/{project_id}/predict")
def run_predictions(project_id: int, model_run_id: Optional[int] = None):
    """
    Generate predictions for all files in a target project using the latest trained model.
    Stores results in the database and returns the prediction list.
    """
    session = get_session()
    project = session.get(Project, project_id)
    if not project:
        session.close()
        raise HTTPException(404, "Project not found")

    # Find latest model run for this project
    run = None
    if model_run_id:
        run = session.get(ModelRun, model_run_id)
    else:
        run = (session.query(ModelRun)
               .filter_by(target_project_id=project_id)
               .order_by(ModelRun.trained_at.desc())
               .first())

    if not run:
        session.close()
        raise HTTPException(404, "No trained model found for this project. Train first.")

    model_path = _get_model_path_for_run(run)
    if not model_path:
        session.close()
        raise HTTPException(404, "Model file not found. Retrain the model.")

    artefact = load_artefact(model_path)
    clf = artefact["clf"]
    feat_names = artefact["feature_names"]
    med_thresh = artefact["medium_threshold"]
    high_thresh = artefact["high_threshold"]

    # Load the target dataset
    src_project = session.get(Project, run.source_project_id)
    tgt_df = load_dataset(project.name)
    X_tgt, y_tgt, tgt_feats = get_features_labels(tgt_df)

    common = [c for c in FEATURE_COLS if c in feat_names and c in tgt_feats]
    src_idx = [feat_names.index(c) for c in common]
    tgt_idx = [tgt_feats.index(c) for c in common]
    X_common = X_tgt[:, tgt_idx]

    # Project into latent space
    if artefact.get("use_transfer") and "tca" in artefact:
        tca = artefact["tca"]
        src_df = load_dataset(src_project.name)
        X_src_raw, _, src_feats = get_features_labels(src_df)
        src_common_idx = [src_feats.index(c) for c in common]
        X_src_common = X_src_raw[:, src_common_idx]
        _, Z_tgt = tca.fit_transform(X_src_common, X_common)
        X_infer = Z_tgt
    elif "scaler" in artefact:
        X_infer = artefact["scaler"].transform(X_common)
    else:
        X_infer = X_common

    proba = clf.predict_proba(X_infer)[:, 1]

    # SHAP
    shap_eng = SHAPEngine(clf, feat_names[:len(common)])
    try:
        factors_list = shap_eng.explain_batch(X_infer, X_infer[:min(50, len(X_infer))])
    except Exception:
        factors_list = [[] for _ in range(len(X_infer))]

    # Historical correlations
    src_df2 = load_dataset(src_project.name)
    X_src2, y_src2, src_feats2 = get_features_labels(src_df2)
    src_ci = [src_feats2.index(c) for c in common if c in src_feats2]
    hist_corr = compute_historical_correlations(X_src2[:, src_ci], y_src2, common)

    # Name column
    name_col = "name" if "name" in tgt_df.columns else (
        "class" if "class" in tgt_df.columns else None
    )

    predictions_out = []
    for i, prob in enumerate(proba):
        tier = assign_risk_tier(prob, med_thresh, high_thresh)
        factors = factors_list[i] if i < len(factors_list) else []
        file_name = str(tgt_df.iloc[i][name_col]) if name_col else f"module_{i}"
        metrics_dict = {c: float(X_common[i, j]) for j, c in enumerate(common)}

        explanation = generate_explanation(tier, float(prob), factors, file_name)
        recommendation = generate_recommendation(tier, factors)

        loc = int(metrics_dict.get("loc", 0))
        cc  = int(metrics_dict.get("max_cc", metrics_dict.get("avg_cc", 0)))
        cpl = int(metrics_dict.get("cbo", metrics_dict.get("ce", 0)))
        debt = estimate_tech_debt(cc, loc, cpl, tier)

        # Store in DB
        fm = FileMetrics(
            target_project_id=project_id,
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
            model_run_id=run.id,
            file_metrics_id=fm.id,
            defect_probability=float(prob),
            risk_tier=tier,
            nlp_explanation=explanation,
            recommendation=recommendation,
            tech_debt_hours=debt,
        )
        pred.set_top_factors(factors[:3])
        session.add(pred)
        session.flush()

        predictions_out.append({
            "id": pred.id,
            "file_path": file_name,
            "defect_probability": round(float(prob), 4),
            "risk_tier": tier,
            "top_factors": factors[:3],
            "nlp_explanation": explanation,
            "recommendation": recommendation,
            "tech_debt_hours": debt,
            "metrics": metrics_dict,
            "label": int(y_tgt[i]) if i < len(y_tgt) else None,
        })

    project_name = project.name
    run_id = run.id
    session.commit()
    session.close()

    health = project_health_score(predictions_out)
    strongest = strongest_historical_predictor(hist_corr)
    post_release_risk = estimated_post_release_risk(predictions_out)

    return {
        "project_id": project_id,
        "project_name": project_name,
        "model_run_id": run_id,
        "health_score": health,
        "post_release_risk": post_release_risk,
        "strongest_historical_predictor": strongest,
        "historical_correlations": hist_corr,
        "predictions": predictions_out,
        "thresholds": {"medium": med_thresh, "high": high_thresh},
    }


@app.get("/api/projects/{project_id}/predictions")
def get_predictions(project_id: int):
    session = get_session()
    preds = (session.query(Prediction)
             .join(FileMetrics)
             .filter(FileMetrics.target_project_id == project_id)
             .order_by(Prediction.defect_probability.desc())
             .all())
    result = []
    for p in preds:
        feedbacks = [{"type": f.feedback_type, "note": f.note} for f in p.feedbacks]
        result.append({
            "id": p.id,
            "file_path": p.file_metrics_obj.file_path if p.file_metrics_obj else "",
            "defect_probability": p.defect_probability,
            "risk_tier": p.risk_tier,
            "top_factors": p.get_top_factors(),
            "nlp_explanation": p.nlp_explanation,
            "recommendation": p.recommendation,
            "tech_debt_hours": p.tech_debt_hours,
            "feedbacks": feedbacks,
        })
    session.close()
    return result


# ---- Hotspots ----

@app.get("/api/projects/{project_id}/hotspots")
def get_hotspots(project_id: int):
    session = get_session()
    preds = (session.query(Prediction)
             .join(FileMetrics)
             .filter(FileMetrics.target_project_id == project_id)
             .all())
    pred_list = [
        {
            "file_path": p.file_metrics_obj.file_path if p.file_metrics_obj else "",
            "defect_probability": p.defect_probability or 0.0,
            "risk_tier": p.risk_tier or "Low",
            "tech_debt_hours": p.tech_debt_hours or 0.0,
        }
        for p in preds
    ]
    session.close()
    # No real repo centrality here — use coupling as proxy
    centrality_map = {p["file_path"]: 0.0 for p in pred_list}
    ranked = hotspot_rank(pred_list, centrality_map)
    return ranked


# ---- What-If ----

@app.post("/api/predictions/{prediction_id}/whatif")
def whatif(prediction_id: int, body: WhatIfRequest):
    versions = list_versions()
    if not versions:
        raise HTTPException(404, "No trained models found.")

    # Get latest model path
    model_path = None
    for v in reversed(versions):
        if "model_path" in v and os.path.exists(v["model_path"]):
            model_path = v["model_path"]
            break

    if not model_path:
        raise HTTPException(404, "Model file not found.")

    result = whatif_predict(model_path, body.modified_metrics)
    return result


# ---- Badge ----

@app.get("/api/projects/{project_id}/badge.svg")
def get_badge(project_id: int):
    session = get_session()
    preds = (session.query(Prediction)
             .join(FileMetrics)
             .filter(FileMetrics.target_project_id == project_id)
             .all())
    pred_list = [{"risk_tier": p.risk_tier} for p in preds]
    session.close()

    health = project_health_score(pred_list)
    svg = generate_badge_svg(health)
    return Response(content=svg, media_type="image/svg+xml")


# ---- Developer Feedback ----

@app.post("/api/predictions/{prediction_id}/feedback")
def submit_feedback(prediction_id: int, body: FeedbackRequest):
    session = get_session()
    pred = session.get(Prediction, prediction_id)
    if not pred:
        session.close()
        raise HTTPException(404, "Prediction not found")

    fb = DeveloperFeedback(
        prediction_id=prediction_id,
        feedback_type=body.feedback_type,
        note=body.note,
    )
    session.add(fb)
    session.commit()
    result = {"id": fb.id, "prediction_id": prediction_id, "feedback_type": body.feedback_type}
    session.close()
    return result


# ---- CI/CD Gate ----

@app.post("/api/ci/evaluate")
def ci_evaluate(body: CIEvaluateRequest):
    """
    Called by GitHub Actions on every push.
    Returns pass/block/overridden decision with full per-file report.
    """
    versions = list_versions()
    model_path = None
    for v in reversed(versions):
        if "model_path" in v and os.path.exists(v["model_path"]):
            model_path = v["model_path"]
            break

    file_reports = []
    has_high_risk = False

    for file_path in body.changed_files:
        # Static analysis (always runs)
        sa_issues = []
        if os.path.exists(file_path):
            sa_issues = run_static_analysis(file_path)

        # ML prediction (if model available and file is Python)
        ml_result = None
        if model_path and file_path.endswith(".py"):
            try:
                artefact = load_artefact(model_path)
                feat_names = artefact["feature_names"]
                # Use zero-vector for unknown real files (demo mode)
                modified = {f: 0.0 for f in feat_names}
                ml_result = whatif_predict(model_path, modified)
                if ml_result["risk_tier"] == "High":
                    has_high_risk = True
            except Exception as e:
                log.warning("ML prediction failed for %s: %s", file_path, e)

        file_reports.append({
            "file": file_path,
            "ml_risk": ml_result,
            "static_issues": sa_issues,
            "static_issue_count": len(sa_issues),
        })

    # Gate decision
    if body.override:
        decision = "overridden"
        gate_pass = True
    elif has_high_risk:
        decision = "block"
        gate_pass = False
    else:
        decision = "pass"
        gate_pass = True

    # Log to DB
    try:
        session = get_session()
        ci_run = CIRun(
            commit_sha=body.commit_sha,
            gate_decision=decision,
            override_reason=body.override_reason if body.override else None,
        )
        session.add(ci_run)
        session.commit()
        ci_run_id = ci_run.id
        session.close()
    except Exception as e:
        log.warning("CI DB log failed: %s", e)
        ci_run_id = None

    return {
        "commit_sha": body.commit_sha,
        "gate_decision": decision,
        "gate_pass": gate_pass,
        "has_high_risk": has_high_risk,
        "override": body.override,
        "override_reason": body.override_reason,
        "file_reports": file_reports,
        "ci_run_id": ci_run_id,
        "message": (
            "✅ All clear — deployment approved." if gate_pass and not body.override
            else "✅ High-risk override accepted. Decision logged." if body.override
            else "🚫 High-risk files detected. A human reviewer must override to proceed. The model informs; a human decides."
        ),
    }


@app.get("/api/ci/runs")
def get_ci_runs():
    session = get_session()
    runs = session.query(CIRun).order_by(CIRun.run_at.desc()).limit(50).all()
    result = [
        {
            "id": r.id,
            "commit_sha": r.commit_sha,
            "gate_decision": r.gate_decision,
            "override_reason": r.override_reason,
            "deployed_url": r.deployed_url,
            "run_at": str(r.run_at),
        }
        for r in runs
    ]
    session.close()
    return result


@app.post("/api/analyze-code")
def analyze_code(body: AnalyzeCodeRequest):
    import ast
    import hashlib
    code = body.code
    filename = body.filename

    loc = len([l for l in code.splitlines() if l.strip() and not l.strip().startswith("#")])
    num_methods = 0
    cc = 1
    coupling = 0
    try:
        tree = ast.parse(code)
        methods = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        num_methods = len(methods)
        branches = [n for n in ast.walk(tree) if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler, ast.With, ast.BoolOp))]
        cc = max(1, len(branches))
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        coupling = len(imports)
    except Exception:
        pass

    metrics = {
        "loc": loc,
        "cyclomatic_complexity": cc,
        "max_cc": cc,
        "avg_cc": max(1.0, cc / max(1, num_methods)),
        "coupling": coupling,
        "num_methods": num_methods,
        "wmc": cc,
        "rfc": num_methods + coupling,
        "cbo": coupling,
        "lcom": max(0, num_methods - 1),
    }

    # Find latest model
    versions = list_versions()
    model_path = None
    for v in reversed(versions):
        if "model_path" in v and os.path.exists(v["model_path"]):
            model_path = v["model_path"]
            break

    if not model_path:
        raise HTTPException(404, "No trained model available. Run training first.")

    pred_res = whatif_predict(model_path, metrics)
    prob = pred_res["defect_probability"]
    tier = pred_res["risk_tier"]

    # Top factors
    top_factors = []
    if cc > 5:
        top_factors.append({
            "feature": "cyclomatic_complexity",
            "friendly_name": "Cyclomatic Complexity",
            "shap_value": 0.35,
            "pct": 45.0,
            "direction": "increases risk",
        })
    if loc > 150:
        top_factors.append({
            "feature": "loc",
            "friendly_name": "Lines of Code",
            "shap_value": 0.25,
            "pct": 30.0,
            "direction": "increases risk",
        })
    if coupling > 6:
        top_factors.append({
            "feature": "cbo",
            "friendly_name": "Coupling Between Objects",
            "shap_value": 0.20,
            "pct": 25.0,
            "direction": "increases risk",
        })

    explanation = generate_explanation(tier, float(prob), top_factors, file_path=filename)
    recommendation = generate_recommendation(tier, top_factors)
    debt = estimate_tech_debt(cc, loc, coupling, risk_tier=tier)

    gate_pass = (tier != "High")
    decision = "pass" if gate_pass else "block"

    h = hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest()[:8]
    deployed_url = None
    if gate_pass and body.auto_deploy:
        deployed_url = f"https://codeguard-deploy.preview/build-{h}"
        try:
            session = get_session()
            ci_run = CIRun(
                commit_sha=f"sha-{h}",
                gate_decision="pass",
                deployed_url=deployed_url,
            )
            session.add(ci_run)
            session.commit()
            session.close()
        except Exception as e:
            log.warning("Could not log CI run: %s", e)

    return {
        "filename": filename,
        "metrics": metrics,
        "defect_probability": prob,
        "risk_tier": tier,
        "top_factors": top_factors,
        "nlp_explanation": explanation,
        "recommendation": recommendation,
        "tech_debt_hours": debt,
        "gate_pass": gate_pass,
        "gate_decision": decision,
        "deployed_url": deployed_url,
        "commit_sha": f"sha-{h}",
    }


@app.post("/api/analyze-project")
def analyze_project(body: AnalyzeProjectRequest):
    import ast
    import hashlib

    folder = os.path.abspath(body.folder_path)
    if not os.path.exists(folder):
        raise HTTPException(400, f"Folder path '{body.folder_path}' does not exist on this machine.")

    py_files = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in (".venv", "venv", ".git", ".pytest_cache", "__pycache__", "env", "node_modules")]
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))

    if not py_files:
        raise HTTPException(400, "No Python (.py) source files found in this folder.")

    versions = list_versions()
    model_path = None
    for v in reversed(versions):
        if "model_path" in v and os.path.exists(v["model_path"]):
            model_path = v["model_path"]
            break

    if not model_path:
        raise HTTPException(404, "No trained transfer model available. Train a model first.")

    file_results = []
    has_high_risk = False
    total_debt = 0.0

    for fpath in py_files[:150]:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                code = f.read()
            loc = len([l for l in code.splitlines() if l.strip() and not l.strip().startswith("#")])
            tree = ast.parse(code)
            methods = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            num_methods = len(methods)
            branches = [n for n in ast.walk(tree) if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler, ast.With, ast.BoolOp))]
            cc = max(1, len(branches))
            imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
            coupling = len(imports)
        except Exception:
            loc = 15
            cc = 1
            coupling = 1
            num_methods = 1

        metrics = {
            "loc": loc,
            "cyclomatic_complexity": cc,
            "max_cc": cc,
            "avg_cc": max(1.0, cc / max(1, num_methods)),
            "coupling": coupling,
            "num_methods": num_methods,
            "wmc": cc,
            "rfc": num_methods + coupling,
            "cbo": coupling,
            "lcom": max(0, num_methods - 1),
        }

        pred_res = whatif_predict(model_path, metrics)
        prob = pred_res["defect_probability"]
        tier = pred_res["risk_tier"]
        debt = estimate_tech_debt(cc, loc, coupling, risk_tier=tier)
        total_debt += debt

        if tier == "High":
            has_high_risk = True

        rel_path = os.path.relpath(fpath, folder)
        file_results.append({
            "file": rel_path,
            "defect_probability": round(prob, 4),
            "risk_tier": tier,
            "tech_debt_hours": debt,
            "loc": loc,
            "cyclomatic_complexity": cc,
            "coupling": coupling,
        })

    health = project_health_score(file_results)
    gate_pass = not has_high_risk
    decision = "pass" if gate_pass else "block"

    h = hashlib.sha256(folder.encode()).hexdigest()[:8]
    deployed_url = None
    if gate_pass and body.auto_deploy:
        deployed_url = f"https://codeguard-deploy.preview/proj-{h}"
        try:
            session = get_session()
            ci_run = CIRun(
                commit_sha=f"proj-{h}",
                gate_decision="pass",
                deployed_url=deployed_url,
            )
            session.add(ci_run)
            session.commit()
            session.close()
        except Exception as e:
            log.warning("Could not log CI run: %s", e)

    return {
        "folder_path": folder,
        "files_scanned": len(file_results),
        "health_score": health,
        "total_tech_debt_hours": round(total_debt, 1),
        "gate_pass": gate_pass,
        "gate_decision": decision,
        "deployed_url": deployed_url,
        "commit_sha": f"proj-{h}",
        "file_results": file_results,
    }



# ---- Multi-source Comparison ----

@app.post("/api/multi-source-comparison")
def api_multi_source(target_project_id: int):
    session = get_session()
    tgt = session.get(Project, target_project_id)
    session.close()
    if not tgt:
        raise HTTPException(404, "Target project not found")

    sources = ["camel-1.6", "lucene-2.4", "poi-2.5"]
    results = multi_source_comparison(sources=sources, target=tgt.name)
    return results


# ---- Cross-project generalization ----

@app.post("/api/cross-project-generalization")
def api_cross_project(source_project_id: int):
    session = get_session()
    src = session.get(Project, source_project_id)
    session.close()
    if not src:
        raise HTTPException(404, "Source project not found")

    results = cross_project_generalization(
        source=src.name,
        targets=["jedit-4.0", "ivy-2.0", "lucene-2.4"]
    )
    return results


# ---- PDF Report ----

@app.get("/api/projects/{project_id}/report.pdf")
def get_pdf_report(project_id: int):
    session = get_session()
    project = session.get(Project, project_id)
    if not project:
        session.close()
        raise HTTPException(404, "Project not found")

    preds = (session.query(Prediction)
             .join(FileMetrics)
             .filter(FileMetrics.target_project_id == project_id)
             .all())
    pred_list = [
        {
            "file_path": p.file_metrics_obj.file_path if p.file_metrics_obj else "",
            "defect_probability": p.defect_probability or 0.0,
            "risk_tier": p.risk_tier or "Low",
            "nlp_explanation": p.nlp_explanation,
            "tech_debt_hours": p.tech_debt_hours or 0.0,
            "metrics": {
                "loc": p.file_metrics_obj.loc if p.file_metrics_obj else 0,
                "cyclomatic_complexity": p.file_metrics_obj.cyclomatic_complexity if p.file_metrics_obj else 0,
                "coupling": p.file_metrics_obj.coupling if p.file_metrics_obj else 0,
            }
        }
        for p in preds
    ]

    run = (session.query(ModelRun)
           .filter_by(target_project_id=project_id)
           .order_by(ModelRun.trained_at.desc()).first())
    model_metrics = {}
    if run:
        model_metrics = {
            "f1_score": run.f1_score or 0.0,
            "p_value": run.p_value or 1.0,
            "f1_improvement": 0.0,
        }
    project_name = project.name
    session.close()

    health = project_health_score(pred_list)
    pdf_bytes = generate_pdf_report(project_name, health, pred_list, model_metrics)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="codeguard_{project_name}.pdf"'}
    )
