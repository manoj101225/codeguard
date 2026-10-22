# CodeGuard — Software Defect Risk Prediction Platform

![Code Health](badge.svg)

> **Transfer Learning · Explainable AI · CI/CD Gate · What-If Simulator**

CodeGuard predicts which code modules are likely to be defective, explains every prediction in plain English, and gates CI/CD deployments automatically — using Transfer Component Analysis+ (TCA+) to transfer knowledge from data-rich source projects to data-scarce target projects.

---

## 🚀 Quick Start (3 commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Seed: download datasets, train model, generate predictions
python seed.py

# 3a. Start the API backend
uvicorn backend.main:app --reload --port 8000

# 3b. In a separate terminal — start the dashboard
streamlit run dashboard/app.py
```

Open **http://localhost:8501** for the dashboard, **http://localhost:8000/docs** for the API.

---

## 🏗️ Architecture

```
codeguard/
├── backend/
│   ├── main.py                   # FastAPI — 15 endpoints
│   ├── ml/
│   │   ├── data_loader.py        # PROMISE dataset downloader + cleaner
│   │   ├── transfer_learning.py  # TCA+ (Nam/Pan/Kim ICSE 2013)
│   │   ├── train.py              # Training pipeline + stat. significance
│   │   ├── metrics_extraction.py # radon + AST + git churn + coverage
│   │   └── whatif.py             # What-If re-inference (no retrain)
│   ├── explainability/
│   │   ├── shap_engine.py        # SHAP TreeExplainer + fallback
│   │   └── nlp_explainer.py      # Template NLP — always works
│   ├── static_analysis/
│   │   └── pylint_runner.py      # Pylint + Bandit + ESLint
│   ├── reports/
│   │   ├── debt_estimator.py     # SQALE debt + health score + hotspots
│   │   └── pdf_generator.py      # ReportLab PDF + SVG badge
│   └── db/
│       ├── models.py             # SQLAlchemy ORM (SQLite)
│       └── schema.sql
├── dashboard/
│   └── app.py                    # Streamlit — 5-tab premium UI
├── tests/
│   └── test_codeguard.py         # pytest suite
├── .github/workflows/
│   └── codeguard.yml             # GitHub Actions CI/CD gate
├── seed.py                       # One-command demo setup
└── requirements.txt
```

---

## 🔬 Core ML Pipeline

| Step | Description |
|------|-------------|
| **Source training** | Random Forest + XGBoost on `camel-1.6` (large, high defect ratio) |
| **TCA+ transfer** | RBF-kernel MMD alignment → shared latent subspace |
| **Fine-tuning** | Combined source + small labeled target data |
| **Risk tiers** | Low/Medium/High thresholds from precision-recall curve (not hardcoded) |
| **Stat. testing** | Paired t-test or Wilcoxon over 10 random splits — p-value reported |

**Default datasets:** Source `camel-1.6`, Target `jedit-4.0`, Multi-source `lucene-2.4`, `poi-2.5`

---

## 📊 Dashboard Features

| Tab | What's inside |
|-----|--------------|
| **Module Risk Table** | Sortable file list with SHAP bar chart, NLP explanation, What-If sliders, feedback button |
| **Hotspot Heatmap** | Treemap by risk × dependency centrality |
| **Multi-Source Comparison** | Bar chart: F1 per source project |
| **Model Version History** | Line chart of F1 across retrains |
| **CI/CD Activity Log** | Pass/block/override history + human override UI |

---

## 🔌 API Reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/projects` | Register source/target project |
| `POST` | `/api/train` | Train with optional TCA+ |
| `GET`  | `/api/runs` | List model runs with F1/p-value |
| `POST` | `/api/projects/{id}/predict` | Generate predictions + SHAP |
| `GET`  | `/api/projects/{id}/predictions` | Fetch stored predictions |
| `GET`  | `/api/projects/{id}/hotspots` | Risk × centrality ranking |
| `POST` | `/api/predictions/{id}/whatif` | Re-inference on modified metrics |
| `GET`  | `/api/projects/{id}/badge.svg` | Quality badge SVG |
| `POST` | `/api/predictions/{id}/feedback` | Submit developer feedback |
| `POST` | `/api/ci/evaluate` | CI/CD gate (called by GitHub Actions) |
| `GET`  | `/api/projects/{id}/report.pdf` | Download PDF quality report |

Interactive docs: **http://localhost:8000/docs**

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🚀 Deployment

### Streamlit Community Cloud (Dashboard)
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → "New app"
3. Set **Main file path** to `dashboard/app.py`
4. Set `CODEGUARD_API` secret to your Render backend URL

### Render (FastAPI Backend)
1. Create a new **Web Service** on [render.com](https://render.com)
2. Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
3. Add `CODEGUARD_API` to GitHub Actions secrets

---

## 🎯 Key Design Decisions

- **Human-in-the-loop gate**: A High-risk block is a default, not a lock. Developers can override with a logged justification. *The model informs; a human decides.*
- **Template NLP fallback**: Plain-English explanations never fail — no external API or model download required.
- **Static analysis is independent**: Pylint/Bandit/ESLint runs on every changed file regardless of ML risk tier — catching bugs that the ML model misses.
- **Thresholds from data**: Low/Medium/High cutoffs are computed from the precision-recall curve on the target split, not hardcoded.
- **Developer feedback loop**: False-positive marking is stored and can down-weight a file's complexity signal in future retrains.

---

## 📚 References

- Nam, J., Pan, S.J., Kim, S. (2013). *Transfer Defect Learning*. ICSE 2013.
- Promise Software Engineering Repository: [http://promise.site.uottawa.ca/SERepository/](http://promise.site.uottawa.ca/SERepository/)
- SHAP: Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
