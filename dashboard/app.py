"""
CodeGuard Dashboard — Streamlit app.

Layout:
  - Sidebar: project selector, train, predict controls
  - Top row: Health / F1 / Gate tiles
  - Tab 1: Module Risk Table (sortable, expandable rows with SHAP, What-If, feedback)
  - Tab 2: Hotspot Heatmap
  - Tab 3: Multi-Source Comparison
  - Tab 4: Model Version History
  - Tab 5: CI/CD Activity Log
"""

import os
import sys
import json
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Make sure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from backend.reports.debt_estimator import project_health_score, estimated_post_release_risk
except ImportError:
    project_health_score = lambda p: 50
    estimated_post_release_risk = lambda p: 0.0

# ---- page config (MUST be first Streamlit call) ----
st.set_page_config(
    page_title="CodeGuard — Defect Risk Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.environ.get("CODEGUARD_API", "http://localhost:8000")

# ============================================================
# CSS — premium dark theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit decorations */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #0a0e1a 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
    border-right: 1px solid #1e2a3a;
}

/* Metric tiles */
.metric-card {
    background: linear-gradient(135deg, #1a2035 0%, #1e2840 100%);
    border: 1px solid #2a3a5c;
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: default;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-delta {
    font-size: 0.8rem;
    font-weight: 500;
}

/* Risk badges */
.badge-high   { background:#ff4d4d22; color:#ff6b6b; border:1px solid #ff4d4d55;
                border-radius:8px; padding:3px 12px; font-weight:600; font-size:0.8rem; }
.badge-medium { background:#ff980022; color:#ffa94d; border:1px solid #ff980055;
                border-radius:8px; padding:3px 12px; font-weight:600; font-size:0.8rem; }
.badge-low    { background:#00c85222; color:#51cf66; border:1px solid #00c85255;
                border-radius:8px; padding:3px 12px; font-weight:600; font-size:0.8rem; }

/* File row cards */
.file-card {
    background: linear-gradient(135deg, #141c2e 0%, #1a2540 100%);
    border: 1px solid #253050;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 6px 0;
    transition: border-color 0.2s ease, background 0.2s ease;
}
.file-card:hover {
    border-color: #3d5a99;
    background: linear-gradient(135deg, #1a2540 0%, #1f2e50 100%);
}
.file-card-high   { border-left: 4px solid #ff4d4d; }
.file-card-medium { border-left: 4px solid #ffa94d; }
.file-card-low    { border-left: 4px solid #51cf66; }

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #94a3b8;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin: 24px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2a3a;
}

/* Explanation box */
.explanation-box {
    background: #0f1929;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #94b8d4;
    margin: 8px 0;
}

/* Gate badge */
.gate-pass   { background:#00c85233; color:#51cf66; border:1px solid #00c85277;
               border-radius:10px; padding:6px 20px; font-weight:700; font-size:1rem; }
.gate-block  { background:#ff4d4d33; color:#ff6b6b; border:1px solid #ff4d4d77;
               border-radius:10px; padding:6px 20px; font-weight:700; font-size:1rem; }

/* Override banner */
.override-banner {
    background: linear-gradient(90deg, #2d1b00, #3d2800);
    border: 1px solid #ff980055;
    border-radius: 10px;
    padding: 12px 18px;
    color: #ffa94d;
    font-size: 0.85rem;
    margin: 8px 0;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-weight: 600;
    color: #64748b;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #60a5fa;
    border-bottom: 2px solid #60a5fa;
}

/* Slider accent */
[data-testid="stSlider"] [data-baseweb="slider"] {
    --color: #60a5fa;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1e3a6e, #2563eb);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: opacity 0.2s;
}
.stButton > button:hover {
    opacity: 0.85;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# API helpers
# ============================================================

def api_get(path: str, timeout: int = 15):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error [{path}]: {e}")
        return None


def api_post(path: str, data: dict, timeout: int = 120):
    try:
        r = requests.post(f"{API_BASE}{path}", json=data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error [{path}]: {e}")
        return None


# ============================================================
# Session state defaults
# ============================================================

for key, default in {
    "selected_project_id": None,
    "predictions": [],
    "model_runs": [],
    "health_score": None,
    "post_release_risk": None,
    "strongest_predictor": "",
    "gate_decision": "—",
    "active_f1": None,
    "f1_improvement": None,
    "training_result": None,
    "expanded_file": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## 🛡️ **CodeGuard**")
    st.markdown("*Software Defect Risk Platform*")
    st.markdown("---")

    projects_data = api_get("/api/projects") or []
    target_projects = [p for p in projects_data if p.get("role") == "target"]
    source_projects = [p for p in projects_data if p.get("role") == "source"]

    st.markdown("### 📁 Project")
    if target_projects:
        proj_names = [p["name"] for p in target_projects]
        default_ix = proj_names.index("jedit-4.0") if "jedit-4.0" in proj_names else 0
        sel_name = st.selectbox("Target Project", proj_names, index=default_ix, key="proj_select")
        sel_proj = next((p for p in target_projects if p["name"] == sel_name), None)
        if sel_proj:
            if st.session_state.selected_project_id != sel_proj["id"]:
                st.session_state.selected_project_id = sel_proj["id"]
                st.session_state.predictions = []
                st.session_state.health_score = None
                st.session_state.post_release_risk = None

            # Automatically populate existing predictions if empty
            if not st.session_state.predictions:
                existing_preds = api_get(f"/api/projects/{sel_proj['id']}/predictions")
                if existing_preds:
                    st.session_state.predictions = existing_preds
                    st.session_state.health_score = project_health_score(existing_preds)
                    st.session_state.post_release_risk = estimated_post_release_risk(existing_preds)
                    st.session_state.gate_decision = "BLOCK" if any(p.get("risk_tier") == "High" for p in existing_preds) else "PASS"
    else:
        st.info("No target projects found. Register one below.")
        st.session_state.selected_project_id = None

    st.markdown("---")
    st.markdown("### ➕ Register Project")
    new_name = st.text_input("Dataset name (e.g. jedit-4.0)", placeholder="jedit-4.0")
    new_role = st.selectbox("Role", ["target", "source"])
    if st.button("Register", key="reg_btn"):
        if new_name:
            r = api_post("/api/projects", {"name": new_name, "role": new_role})
            if r:
                st.success(f"Registered: {r['name']} (ID {r['id']})")
                time.sleep(1)
                st.rerun()
        else:
            st.warning("Enter a dataset name.")

    st.markdown("---")
    st.markdown("### 🏋️ Train Model")

    src_names = [p["name"] for p in source_projects] if source_projects else ["camel-1.6"]
    sel_src = st.selectbox("Source project", src_names if src_names else ["camel-1.6"])
    use_tl = st.toggle("Use Transfer Learning (TCA+)", value=True)
    model_type = st.selectbox("Model type", ["rf", "xgb"])

    if st.button("🚀 Train Now", key="train_btn"):
        src_proj = next((p for p in source_projects if p["name"] == sel_src), None)
        if not src_proj:
            # Auto-register source project
            r_s = api_post("/api/projects", {"name": sel_src, "role": "source"})
            src_proj = r_s
        tgt_id = st.session_state.selected_project_id
        if not tgt_id:
            st.error("Select or register a target project first.")
        else:
            with st.spinner("Training… (~60 s for 10 folds)"):
                result = api_post("/api/train", {
                    "source_project_id": src_proj["id"],
                    "target_project_id": tgt_id,
                    "use_transfer": use_tl,
                    "model_type": model_type,
                })
            if result:
                st.session_state.training_result = result
                st.session_state.active_f1 = result.get("f1_score")
                st.session_state.f1_improvement = result.get("f1_improvement")
                st.success(f"✅ F1: {result.get('f1_score', 0):.1%}")

    st.markdown("---")
    st.markdown("### 🔍 Run Predictions")
    if st.button("📊 Generate Predictions", key="pred_btn"):
        pid = st.session_state.selected_project_id
        if not pid:
            st.error("Select a target project first.")
        else:
            with st.spinner("Running predictions + SHAP…"):
                result = api_post(f"/api/projects/{pid}/predict", {})
            if result:
                st.session_state.predictions = result.get("predictions", [])
                st.session_state.health_score = result.get("health_score")
                st.session_state.post_release_risk = result.get("post_release_risk")
                st.session_state.strongest_predictor = result.get("strongest_historical_predictor", "")
                preds = st.session_state.predictions
                if any(p.get("risk_tier") == "High" for p in preds):
                    st.session_state.gate_decision = "BLOCK"
                else:
                    st.session_state.gate_decision = "PASS"
                st.success(f"✅ {len(preds)} predictions ready")

    st.markdown("---")
    st.caption("CodeGuard v1.0 · The model informs; a human decides.")


# ============================================================
# Main area
# ============================================================

st.markdown("<h1 style='font-size:2rem; font-weight:800; margin-bottom:4px;'>🛡️ CodeGuard</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748b; margin-top:0;'>Software Defect Risk Prediction · Transfer Learning · Explainable AI</p>", unsafe_allow_html=True)

preds      = st.session_state.predictions
runs       = api_get("/api/runs") or []
health     = st.session_state.health_score
f1_val     = st.session_state.active_f1
f1_delta   = st.session_state.f1_improvement
gate       = st.session_state.gate_decision
post_risk  = st.session_state.post_release_risk

# If we have predictions but no stored runs, refresh
if preds and not runs:
    runs = api_get("/api/runs") or []
if runs and f1_val is None:
    latest = runs[0]
    f1_val  = latest.get("f1_score")

# ---- Top summary tiles ----
c1, c2, c3, c4 = st.columns(4)

health_color = "#51cf66" if (health or 0) >= 75 else "#ffa94d" if (health or 0) >= 50 else "#ff6b6b"
with c1:
    h = health or "—"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Project Health</div>
        <div class="metric-value" style="color:{health_color}">{h}<span style="font-size:1.2rem;color:#64748b">/100</span></div>
        <div class="metric-delta" style="color:#64748b">{len(preds)} modules analysed</div>
    </div>""", unsafe_allow_html=True)

f1_color = "#51cf66" if (f1_val or 0) >= 0.75 else "#ffa94d"
with c2:
    f1_disp = f"{f1_val:.1%}" if f1_val else "—"
    delta_disp = f"+{f1_delta:.1%} vs no-transfer" if f1_delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Model F1-Score</div>
        <div class="metric-value" style="color:{f1_color}">{f1_disp}</div>
        <div class="metric-delta" style="color:#51cf66">{delta_disp}</div>
    </div>""", unsafe_allow_html=True)

gate_color = "#51cf66" if gate == "PASS" else "#ff6b6b"
gate_icon  = "✅" if gate == "PASS" else "🚫"
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">CI/CD Gate Decision</div>
        <div class="metric-value" style="color:{gate_color}; font-size:1.8rem;">{gate_icon} {gate}</div>
        <div class="metric-delta" style="color:#64748b">Human override available</div>
    </div>""", unsafe_allow_html=True)

with c4:
    pr = f"{post_risk:.2f}" if post_risk is not None else "—"
    high_count = sum(1 for p in preds if p.get("risk_tier") == "High")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Post-Release Risk</div>
        <div class="metric-value" style="color:#f59e0b">{pr}</div>
        <div class="metric-delta" style="color:#64748b">{high_count} High-risk files</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Strongest historical predictor banner
if st.session_state.strongest_predictor:
    st.info(f"📊 {st.session_state.strongest_predictor}", icon="📊")

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Module Risk Table",
    "🔥 Hotspot Heatmap",
    "🚀 Scan File & Auto-Deploy",
    "📊 Multi-Source Comparison",
    "📈 Model Version History",
    "⚙️ CI/CD Activity Log",
])


# ========== TAB 1: Module Risk Table ==========
with tab1:
    if not preds:
        st.markdown("""
        <div style='text-align:center; padding:60px; color:#475569;'>
            <div style='font-size:3rem;'>🔍</div>
            <h3>No predictions yet</h3>
            <p>Register a target project, train a model, and click <b>Generate Predictions</b> in the sidebar.</p>
        </div>""", unsafe_allow_html=True)
    else:
        # Filter / sort controls
        fc1, fc2, fc3 = st.columns([2, 2, 2])
        with fc1:
            tier_filter = st.multiselect(
                "Filter by Risk Tier",
                ["High", "Medium", "Low"],
                default=["High", "Medium", "Low"]
            )
        with fc2:
            sort_by = st.selectbox("Sort by", ["Risk Score ↓", "Debt Hours ↓", "File Name ↑"])
        with fc3:
            search_term = st.text_input("🔍 Search files", placeholder="payment_service…")

        # Apply filters
        filtered = [p for p in preds if p.get("risk_tier") in tier_filter]
        if search_term:
            filtered = [p for p in filtered if search_term.lower() in (p.get("file_path") or "").lower()]
        if sort_by == "Risk Score ↓":
            filtered = sorted(filtered, key=lambda p: p.get("defect_probability", 0), reverse=True)
        elif sort_by == "Debt Hours ↓":
            filtered = sorted(filtered, key=lambda p: p.get("tech_debt_hours", 0), reverse=True)
        else:
            filtered = sorted(filtered, key=lambda p: (p.get("file_path") or ""))

        st.markdown(f"<div class='section-header'>Files ({len(filtered)} shown)</div>", unsafe_allow_html=True)

        tier_emoji = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
        tier_class = {"High": "file-card-high", "Medium": "file-card-medium", "Low": "file-card-low"}

        for pred in filtered:
            tier  = pred.get("risk_tier", "Low")
            prob  = pred.get("defect_probability", 0.0)
            fname = pred.get("file_path", "—")
            metrics = pred.get("metrics", {})
            loc  = int(metrics.get("loc", 0))
            cc   = int(metrics.get("max_cc", metrics.get("avg_cc", metrics.get("cyclomatic_complexity", 0))))
            cpl  = int(metrics.get("cbo", metrics.get("ce", metrics.get("coupling", 0))))
            debt = pred.get("tech_debt_hours", 0.0)
            pred_id = pred.get("id", 0)

            # Row header (always visible)
            col_a, col_b, col_c = st.columns([6, 3, 2])
            with col_a:
                st.markdown(f"""
                <div class="file-card {tier_class.get(tier, '')}">
                    <span style='font-weight:700; color:#e2e8f0; font-size:0.95rem;'>{fname}</span>
                    <span style='color:#475569; font-size:0.82rem; margin-left:12px;'>
                        LOC {loc} · CC {cc} · Coupling {cpl}
                    </span>
                </div>""", unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div style='padding:14px 0;'>
                    <span class='badge-{tier.lower()}'>{tier_emoji.get(tier,'')} {tier}</span>
                    <span style='color:#64748b; font-size:0.82rem; margin-left:10px;'>{prob:.1%}</span>
                </div>""", unsafe_allow_html=True)
            with col_c:
                btn_label = f"{'▼' if st.session_state.expanded_file == pred_id else '▶'} Details"
                if st.button(btn_label, key=f"expand_{pred_id}"):
                    if st.session_state.expanded_file == pred_id:
                        st.session_state.expanded_file = None
                    else:
                        st.session_state.expanded_file = pred_id

            # Expanded detail panel
            if st.session_state.expanded_file == pred_id:
                with st.container():
                    st.markdown("---")
                    d1, d2 = st.columns([1, 1])

                    with d1:
                        st.markdown("**🧠 SHAP Feature Attribution**")
                        factors = pred.get("top_factors", [])
                        if factors:
                            fnames_shap = [f.get("friendly_name", f.get("feature", "")) for f in factors]
                            pcts  = [f.get("pct", 0) for f in factors]
                            svs   = [f.get("shap_value", 0) for f in factors]
                            colors_shap = ["#ef4444" if v >= 0 else "#3b82f6" for v in svs]

                            fig = go.Figure(go.Bar(
                                x=pcts,
                                y=fnames_shap,
                                orientation="h",
                                marker_color=colors_shap,
                                text=[f"{p:.0f}%" for p in pcts],
                                textposition="outside",
                            ))
                            fig.update_layout(
                                height=200,
                                margin=dict(l=0, r=40, t=10, b=10),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#94a3b8", size=11),
                                xaxis=dict(showgrid=False, zeroline=False),
                                yaxis=dict(showgrid=False),
                                showlegend=False,
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"shap_{pred_id}")
                        else:
                            st.info("No SHAP data available.")

                        explanation = pred.get("nlp_explanation", "")
                        if explanation:
                            st.markdown(f"""
                            <div class='explanation-box'>
                                💬 {explanation}
                            </div>""", unsafe_allow_html=True)

                        rec = pred.get("recommendation", "")
                        if rec:
                            st.markdown(f"""
                            <div class='explanation-box' style='border-color:#1a4a2e;color:#6ee7b7;'>
                                🔧 <b>Recommendation:</b> {rec}
                            </div>""", unsafe_allow_html=True)

                        st.markdown(f"💳 **Technical Debt Estimate:** ~{debt} hours to fix")

                    with d2:
                        st.markdown("**🔬 What-If Risk Simulator**")
                        st.caption("Edit metrics and see how the risk score changes — no retraining needed.")

                        wi_loc = st.slider("LOC", 0, 2000, min(loc, 2000), 10,
                                           key=f"wi_loc_{pred_id}")
                        wi_cc  = st.slider("Cyclomatic Complexity", 0, 100,
                                           min(cc, 100), 1, key=f"wi_cc_{pred_id}")
                        wi_cpl = st.slider("Coupling", 0, 50,
                                           min(cpl, 50), 1, key=f"wi_cpl_{pred_id}")
                        wi_npm = st.slider("Number of Methods", 0, 100,
                                           min(int(metrics.get("npm", 5)), 100), 1,
                                           key=f"wi_npm_{pred_id}")

                        if st.button("⚡ Simulate", key=f"sim_{pred_id}"):
                            with st.spinner("Re-running inference…"):
                                # Build modified metrics using known feature names
                                modified = dict(metrics)
                                modified.update({
                                    "loc": wi_loc, "max_cc": wi_cc, "avg_cc": wi_cc,
                                    "cbo": wi_cpl, "ce": wi_cpl,
                                    "npm": wi_npm,
                                    "wmc": wi_npm,
                                })
                                result = api_post(
                                    f"/api/predictions/{pred_id}/whatif",
                                    {"model_run_id": 0, "modified_metrics": modified}
                                )
                            if result:
                                new_prob = result.get("defect_probability", 0)
                                new_tier = result.get("risk_tier", "—")
                                delta_prob = new_prob - prob
                                delta_color = "#51cf66" if delta_prob < 0 else "#ff6b6b"
                                delta_sign  = "▼" if delta_prob < 0 else "▲"
                                st.markdown(f"""
                                <div style='background:#0f1929;border:1px solid #1e3a5f;
                                            border-radius:10px;padding:16px;margin-top:8px;'>
                                    <div style='font-size:0.8rem;color:#64748b;'>New Risk Score</div>
                                    <div style='font-size:2rem;font-weight:800;
                                                color:{gate_color if new_tier=="PASS" else "#ff6b6b"};'>
                                        {new_prob:.1%}
                                    </div>
                                    <div style='font-size:0.9rem;'>
                                        Tier: <b>{new_tier}</b> &nbsp;
                                        <span style='color:{delta_color};'>
                                            {delta_sign} {abs(delta_prob):.1%} vs original
                                        </span>
                                    </div>
                                </div>""", unsafe_allow_html=True)

                        st.markdown("---")
                        st.markdown("**🏷️ Developer Feedback**")
                        fb_type = st.selectbox(
                            "Feedback type",
                            ["confirmed", "false_positive", "essential_complexity", "needs_refactor"],
                            key=f"fb_type_{pred_id}"
                        )
                        fb_note = st.text_input("Note (optional)", key=f"fb_note_{pred_id}",
                                                placeholder="This complexity is necessary because…")
                        if st.button("Submit Feedback", key=f"fb_btn_{pred_id}"):
                            r = api_post(f"/api/predictions/{pred_id}/feedback",
                                         {"feedback_type": fb_type, "note": fb_note})
                            if r:
                                if fb_type == "false_positive":
                                    st.success("✅ Marked as false positive. Badge updated.")
                                else:
                                    st.success("✅ Feedback recorded.")
                    st.markdown("---")


# ========== TAB 2: Hotspot Heatmap ==========
with tab2:
    st.markdown("### 🔥 Hotspot Heatmap")
    st.caption("Risk score × dependency centrality — risky files that many others depend on rank highest.")

    pid = st.session_state.selected_project_id
    if pid:
        hotspots = api_get(f"/api/projects/{pid}/hotspots") or []
    else:
        hotspots = []

    if preds and not hotspots:
        hotspots = sorted(preds, key=lambda p: p.get("defect_probability", 0), reverse=True)

    if hotspots:
        df_hs = pd.DataFrame(hotspots[:40])
        df_hs["file_label"] = df_hs.get("file_path", pd.Series(dtype=str)).apply(
            lambda x: (x or "")[-40:] if x else ""
        )
        prob_col = "defect_probability" if "defect_probability" in df_hs else "hotspot_score"
        tier_col = "risk_tier" if "risk_tier" in df_hs else None

        color_map = {"High": "#ef4444", "Medium": "#f97316", "Low": "#22c55e"}

        fig_hs = px.treemap(
            df_hs,
            path=["file_label"],
            values=prob_col,
            color=tier_col if tier_col else prob_col,
            color_discrete_map=color_map if tier_col else None,
            color_continuous_scale=["#22c55e", "#f97316", "#ef4444"] if not tier_col else None,
            title="",
        )
        fig_hs.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            height=420,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        fig_hs.update_traces(textfont_size=12)
        st.plotly_chart(fig_hs, use_container_width=True, key="hotspot_tree")

        # Ranked list
        st.markdown("**Top 10 Hotspots**")
        for i, h in enumerate(hotspots[:10], 1):
            tier = h.get("risk_tier", "Low")
            score = h.get("hotspot_score", h.get("defect_probability", 0))
            debt  = h.get("tech_debt_hours", 0)
            fname = (h.get("file_path") or "")[-60:]
            st.markdown(f"""
            <div style='padding:8px 14px; margin:4px 0; border-radius:8px;
                        background:#141c2e; border-left:3px solid
                        {"#ef4444" if tier=="High" else "#f97316" if tier=="Medium" else "#22c55e"};
                        display:flex; justify-content:space-between;'>
                <span style='color:#e2e8f0;'><b>#{i}</b> {fname}</span>
                <span style='color:#94a3b8; font-size:0.82rem;'>
                    Hotspot: {score:.3f} · Debt: {debt}h · {tier}
                </span>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Run predictions to see the hotspot heatmap.")


# ========== TAB 3: Scan File & Auto-Deploy ==========
with tab3:
    st.markdown("### 🚀 Real-time File Risk Scanner & Automated Deployment Gate")
    st.markdown(
        "Upload or paste any Python source file. CodeGuard extracts its code metrics (AST complexity, coupling, LOC), "
        "compares them against the **Transfer Learning model** (trained on historical defect benchmarks), "
        "and automatically approves or blocks deployment."
    )

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        scan_mode = st.radio("Scan Target", ["📂 Entire Laptop Project Folder", "📁 Upload Single File", "📝 Paste Code Snippet"], horizontal=True)
        filename_input = "custom_module.py"
        code_text = ""
        folder_input = ""

        if scan_mode == "📂 Entire Laptop Project Folder":
            st.markdown("Scan all Python files in any directory on your computer:")
            folder_input = st.text_input(
                "Local Folder Path",
                value=os.path.abspath("backend"),
                help="Enter an absolute path, e.g. C:\\Users\\gajul\\OneDrive\\Desktop\\my_project"
            )
        elif scan_mode == "📁 Upload Single File":
            uploaded_file = st.file_uploader("Upload Python module (.py)", type=["py", "txt", "js"])
            if uploaded_file is not None:
                filename_input = uploaded_file.name
                code_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
                st.caption(f"Uploaded: **{filename_input}** ({len(code_text.splitlines())} lines)")
        else:
            filename_input = st.text_input("Module Filename", value="payment_service.py")
            default_sample = '''import os
import requests

def process_transaction(user_id, amount, currency="USD"):
    """Process incoming payment transaction."""
    if amount <= 0:
        raise ValueError("Invalid amount")
    
    if currency not in ["USD", "EUR", "GBP"]:
        return {"status": "error", "reason": "unsupported_currency"}
        
    retries = 3
    for attempt in range(retries):
        try:
            resp = requests.post("https://api.payment.internal/charge", json={"uid": user_id, "amt": amount})
            if resp.status_code == 200:
                return {"status": "success", "txid": resp.json().get("id")}
        except Exception:
            continue
            
    return {"status": "failed", "reason": "gateway_timeout"}
'''
            code_text = st.text_area("Paste Python Code", value=default_sample, height=260)

        auto_deploy_check = st.checkbox("Automatically trigger deployment if gate passes (PASS)", value=True)
        scan_btn = st.button("⚡ Scan & Check Gate", type="primary", use_container_width=True)

    with col_result:
        if scan_btn:
            if scan_mode == "📂 Entire Laptop Project Folder":
                if not folder_input.strip():
                    st.warning("Please provide a local folder path.")
                else:
                    with st.spinner(f"Scanning all modules in '{folder_input}' and comparing with PROMISE models..."):
                        proj_res = api_post("/api/analyze-project", {
                            "folder_path": folder_input,
                            "auto_deploy": auto_deploy_check,
                        })

                    if proj_res:
                        p_health = proj_res.get("health_score", 0)
                        p_gate = proj_res.get("gate_pass", True)
                        p_url = proj_res.get("deployed_url")
                        p_debt = proj_res.get("total_tech_debt_hours", 0)
                        p_scanned = proj_res.get("files_scanned", 0)
                        p_files = proj_res.get("file_results", [])

                        gate_color = "#51cf66" if p_gate else "#ff6b6b"
                        gate_title = "✅ ENTIRE PROJECT DEPLOYMENT APPROVED" if p_gate else "🚫 PROJECT DEPLOYMENT BLOCKED"
                        gate_bg = "rgba(81, 207, 102, 0.15)" if p_gate else "rgba(255, 107, 107, 0.15)"

                        st.markdown(f"""
                        <div style="border: 1px solid {gate_color}; background: {gate_bg}; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                            <h3 style="color: {gate_color}; margin: 0 0 8px 0;">{gate_title}</h3>
                            <p style="color: #cbd5e1; margin: 0; font-size: 1.05rem;">
                                Project Health: <b>{p_health}/100</b> &nbsp;|&nbsp; Files Scanned: <b>{p_scanned}</b> &nbsp;|&nbsp; Tech Debt: <b>{p_debt}h</b>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        if p_url:
                            st.success("🌐 **Live Project Deployment Link Ready!**")
                            st.markdown(f"""
                            <div style="background: #1e293b; padding: 14px 18px; border-radius: 8px; border-left: 4px solid #38bdf8; margin-bottom: 16px;">
                                <span style="color: #94a3b8; font-size: 0.85rem;">PROJECT PREVIEW URL:</span><br>
                                <a href="{p_url}" target="_blank" style="color: #38bdf8; font-weight: 600; font-size: 1.1rem; text-decoration: none;">
                                    {p_url} 🔗
                                </a>
                                <div style="color: #64748b; font-size: 0.8rem; margin-top: 4px;">Commit SHA: <code>{proj_res.get('commit_sha')}</code> · Logged to CI/CD Activity</div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("#### 📋 Scanned Modules Breakdown")
                        df_res = pd.DataFrame(p_files)
                        if not df_res.empty:
                            st.dataframe(
                                df_res[["file", "risk_tier", "defect_probability", "tech_debt_hours", "loc", "cyclomatic_complexity"]].rename(columns={
                                    "file": "File Path",
                                    "risk_tier": "Risk Tier",
                                    "defect_probability": "Defect Prob",
                                    "tech_debt_hours": "Debt (h)",
                                    "loc": "LOC",
                                    "cyclomatic_complexity": "CC"
                                }),
                                use_container_width=True,
                                height=280
                            )
            else:
                if not code_text.strip():
                    st.warning("Please provide code content to scan.")
                else:
                    with st.spinner("Extracting AST metrics & evaluating defect probability..."):
                        res = api_post("/api/analyze-code", {
                            "filename": filename_input,
                            "code": code_text,
                            "auto_deploy": auto_deploy_check
                        })

                    if res:
                        tier = res.get("risk_tier", "Low")
                        prob = res.get("defect_probability", 0.0)
                        gate_pass = res.get("gate_pass", True)
                        deployed_url = res.get("deployed_url")
                        m = res.get("metrics", {})

                        tier_color = "#51cf66" if tier == "Low" else "#ffa94d" if tier == "Medium" else "#ff6b6b"
                        gate_badge = "✅ DEPLOYMENT APPROVED (PASS)" if gate_pass else "🚫 DEPLOYMENT BLOCKED (HIGH RISK)"
                        gate_bg = "rgba(81, 207, 102, 0.15)" if gate_pass else "rgba(255, 107, 107, 0.15)"
                        gate_border = "#51cf66" if gate_pass else "#ff6b6b"

                        st.markdown(f"""
                        <div style="border: 1px solid {gate_border}; background: {gate_bg}; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                            <h3 style="color: {tier_color}; margin: 0 0 8px 0;">{gate_badge}</h3>
                            <p style="color: #cbd5e1; margin: 0; font-size: 1.05rem;">
                                Defect Risk: <b>{prob:.1%}</b> &nbsp;|&nbsp; Risk Tier: <b>{tier}</b> &nbsp;|&nbsp; Tech Debt: <b>{res.get('tech_debt_hours', 0)}h</b>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        if deployed_url:
                            st.success("🌐 **Live Deployment Link Generated!**")
                            st.markdown(f"""
                            <div style="background: #1e293b; padding: 14px 18px; border-radius: 8px; border-left: 4px solid #38bdf8; margin-bottom: 16px;">
                                <span style="color: #94a3b8; font-size: 0.85rem;">DEPLOYED PREVIEW URL:</span><br>
                                <a href="{deployed_url}" target="_blank" style="color: #38bdf8; font-weight: 600; font-size: 1.1rem; text-decoration: none;">
                                    {deployed_url} 🔗
                                </a>
                                <div style="color: #64748b; font-size: 0.8rem; margin-top: 4px;">Commit SHA: <code>{res.get('commit_sha')}</code> · Recorded in CI/CD Activity Log</div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif not gate_pass:
                            st.error("🚫 This module was blocked because it poses high defect risk to production. Refactor the items below or request a manual manager override.")

                        st.markdown("#### 🧠 AI Explanation & Recommendation")
                        st.info(res.get("nlp_explanation", ""))
                        if res.get("recommendation"):
                            st.warning(f"💡 **Recommendation:** {res.get('recommendation')}")

                        st.markdown("#### 📊 Extracted Code Metrics")
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("LOC", m.get("loc", 0))
                        mc2.metric("Cyclomatic Complexity", m.get("cyclomatic_complexity", 0))
                        mc3.metric("Dependencies (Coupling)", m.get("coupling", 0))
                        mc4.metric("Methods Count", m.get("num_methods", 0))
        else:
            st.markdown("""
            <div style="border: 2px dashed #334155; border-radius: 12px; padding: 40px; text-align: center; color: #64748b;">
                <div style="font-size: 2.5rem; margin-bottom: 10px;">⚡</div>
                <h4>Choose an option on the left</h4>
                <p>Select <b>Entire Laptop Project Folder</b> to scan a whole folder, or upload a single file to test.</p>
            </div>
            """, unsafe_allow_html=True)


# ========== TAB 4: Multi-Source Comparison ==========
with tab4:
    st.markdown("### 📊 Multi-Source Comparison")
    st.caption("Which source project transfers best to your target? Higher F1 = better knowledge transfer.")

    pid = st.session_state.selected_project_id
    if st.button("▶ Run Multi-Source Study", key="msc_btn") and pid:
        with st.spinner("Training with 3 source projects… (~3 min)"):
            results = api_post(f"/api/multi-source-comparison?target_project_id={pid}", {})
        if results:
            st.session_state["msc_results"] = results

    msc = st.session_state.get("msc_results", [])
    if runs:
        # Add current model runs as comparison data
        msc_from_runs = []
        for r in runs[:5]:
            if r.get("f1_score"):
                msc_from_runs.append({
                    "source": r.get("source_project", "unknown"),
                    "target": r.get("target_project", ""),
                    "f1": r["f1_score"],
                    "used_tl": r.get("used_transfer_learning", True),
                })
        if msc_from_runs and not msc:
            msc = msc_from_runs

    if msc:
        df_msc = pd.DataFrame(msc)
        if "f1" in df_msc.columns:
            df_msc = df_msc.dropna(subset=["f1"])
            df_msc["f1_pct"] = df_msc["f1"] * 100

            colors_msc = px.colors.qualitative.Set2[:len(df_msc)]
            fig_msc = go.Figure(go.Bar(
                x=df_msc["source"],
                y=df_msc["f1_pct"],
                marker_color=colors_msc,
                text=[f"{v:.1f}%" for v in df_msc["f1_pct"]],
                textposition="outside",
            ))
            fig_msc.update_layout(
                xaxis_title="Source Project",
                yaxis_title="F1-Score (%)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                height=380,
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis=dict(range=[0, 105], gridcolor="#1e2a3a"),
                xaxis=dict(gridcolor="#1e2a3a"),
            )
            st.plotly_chart(fig_msc, use_container_width=True, key="msc_bar")

            best = df_msc.loc[df_msc["f1"].idxmax()]
            st.success(f"🏆 Best source: **{best['source']}** (F1 = {best['f1']:.1%})")

            # With vs Without transfer comparison
            tl_runs  = [r for r in runs if r.get("used_transfer_learning") and r.get("f1_score")]
            base_runs = [r for r in runs if not r.get("used_transfer_learning") and r.get("f1_score")]
            if tl_runs and base_runs:
                tl_f1   = tl_runs[0]["f1_score"]
                base_f1 = base_runs[0]["f1_score"]
                imp = (tl_f1 - base_f1) * 100
                c1_, c2_ = st.columns(2)
                c1_.metric("With Transfer Learning", f"{tl_f1:.1%}")
                c2_.metric("Without Transfer (baseline)", f"{base_f1:.1%}",
                           delta=f"{imp:+.1f}%")

                p_vals = [r.get("p_value") for r in tl_runs if r.get("p_value") is not None]
                if p_vals:
                    p = p_vals[0]
                    sig = "✅ statistically significant (p < 0.05)" if p < 0.05 else "⚠️ not significant at p < 0.05"
                    st.markdown(f"**Statistical test:** p = {p:.4f} — {sig}")
    else:
        st.info("Click 'Run Multi-Source Study' or train multiple models to see the comparison.")


# ========== TAB 5: Model Version History ==========
with tab5:
    st.markdown("### 📈 Model Version History")
    st.caption("Every retrain is logged — track how the model improves over time.")

    versions = api_get("/api/model-versions") or []
    if not versions and runs:
        # Reconstruct from DB runs
        versions = [
            {
                "source": r.get("source_project"),
                "target": r.get("target_project"),
                "f1_score": r.get("f1_score"),
                "trained_at": r.get("trained_at"),
                "use_transfer": r.get("used_transfer_learning"),
                "p_value": r.get("p_value"),
            }
            for r in runs if r.get("f1_score")
        ]

    if versions:
        df_v = pd.DataFrame(versions)
        if "f1_score" in df_v.columns and "trained_at" in df_v.columns:
            df_v = df_v.dropna(subset=["f1_score"])
            df_v["trained_at"] = pd.to_datetime(df_v["trained_at"])
            df_v = df_v.sort_values("trained_at")
            df_v["f1_pct"] = df_v["f1_score"] * 100
            df_v["label"] = df_v["source"].astype(str) + " → " + df_v["target"].astype(str)

            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(
                x=df_v["trained_at"],
                y=df_v["f1_pct"],
                mode="lines+markers+text",
                text=[f"{v:.1f}%" for v in df_v["f1_pct"]],
                textposition="top center",
                line=dict(color="#60a5fa", width=2),
                marker=dict(size=10, color="#60a5fa",
                            line=dict(color="#1e3a6e", width=2)),
                name="F1-Score",
            ))
            fig_v.update_layout(
                xaxis_title="Training Time",
                yaxis_title="F1-Score (%)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                height=340,
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis=dict(range=[0, 105], gridcolor="#1e2a3a"),
                xaxis=dict(gridcolor="#1e2a3a"),
            )
            st.plotly_chart(fig_v, use_container_width=True, key="version_line")

        # Version table
        st.markdown("**All Runs**")
        display_cols = ["trained_at", "source", "target", "f1_score",
                        "use_transfer", "p_value"]
        avail = [c for c in display_cols if c in df_v.columns]
        st.dataframe(
            df_v[avail].rename(columns={
                "trained_at": "Trained At",
                "source": "Source",
                "target": "Target",
                "f1_score": "F1-Score",
                "use_transfer": "Transfer?",
                "p_value": "p-value",
            }).style.format({
                "F1-Score": "{:.3f}",
                "p-value": "{:.4f}",
            }),
            use_container_width=True,
            key="version_table"
        )
    else:
        st.info("Train a model to start tracking version history.")


# ========== TAB 6: CI/CD Activity Log ==========
with tab6:
    st.markdown("### ⚙️ CI/CD Activity Log")
    st.caption("Recent commits evaluated by the CodeGuard gate. The model informs; a human decides.")

    ci_runs = api_get("/api/ci/runs") or []

    if ci_runs:
        for run in ci_runs[:20]:
            decision = run.get("gate_decision", "—")
            sha      = (run.get("commit_sha") or "—")[:10]
            ts       = run.get("run_at", "—")
            override_reason = run.get("override_reason")

            dec_color = {"pass": "#51cf66", "block": "#ff6b6b", "overridden": "#ffa94d"}.get(decision, "#94a3b8")
            dec_icon  = {"pass": "✅", "block": "🚫", "overridden": "⚠️"}.get(decision, "❓")

            st.markdown(f"""
            <div style='padding:12px 18px; margin:4px 0; border-radius:10px;
                        background:#141c2e; border:1px solid #253050;
                        display:flex; align-items:center; gap:16px;'>
                <span style='font-family:monospace; color:#94a3b8; font-size:0.85rem;'>{sha}</span>
                <span style='color:{dec_color}; font-weight:700;'>{dec_icon} {decision.upper()}</span>
                <span style='color:#64748b; font-size:0.82rem;'>{ts}</span>
                {"<span style='color:#ffa94d; font-size:0.82rem;'>Override: " + override_reason + "</span>" if override_reason else ""}
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No CI runs yet. Use `POST /api/ci/evaluate` from GitHub Actions.")

    # Human override demo
    st.markdown("---")
    st.markdown("**🧑‍💻 Human Override Demo**")
    st.markdown("""
    <div class='override-banner'>
        ⚠️ <b>CodeGuard's philosophy:</b> A High-risk block is a default, not an absolute lock.
        A developer can override with a one-line justification — which is logged permanently.
        <i>The model informs; a human decides.</i>
    </div>""", unsafe_allow_html=True)

    col_o1, col_o2 = st.columns([3, 1])
    with col_o1:
        override_note = st.text_input(
            "Override justification",
            placeholder="This complexity is intentional — it handles edge cases in the payment flow.",
            key="override_note"
        )
    with col_o2:
        if st.button("🔓 Override & Merge", key="override_btn"):
            if override_note.strip():
                result = api_post("/api/ci/evaluate", {
                    "commit_sha": "manual-override",
                    "changed_files": [],
                    "override": True,
                    "override_reason": override_note,
                })
                if result:
                    st.success(f"✅ Override logged: {result.get('message', '')}")
            else:
                st.warning("Enter a justification — it will be logged permanently.")
