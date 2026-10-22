-- CodeGuard Database Schema (SQLite-compatible)
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(150) NOT NULL,
    role VARCHAR(20) CHECK (role IN ('source','target')),
    repo_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_project_id INT REFERENCES projects(id),
    target_project_id INT REFERENCES projects(id),
    used_transfer_learning BOOLEAN,
    f1_score FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    p_value FLOAT,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS file_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_project_id INT REFERENCES projects(id),
    file_path TEXT NOT NULL,
    loc INT,
    cyclomatic_complexity INT,
    coupling INT,
    num_methods INT,
    churn_count INT,
    test_coverage_pct FLOAT,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_run_id INT REFERENCES model_runs(id),
    file_metrics_id INT REFERENCES file_metrics(id),
    defect_probability FLOAT,
    risk_tier VARCHAR(10) CHECK (risk_tier IN ('Low','Medium','High')),
    top_factors TEXT,       -- JSON stored as text in SQLite
    nlp_explanation TEXT,
    recommendation TEXT,
    tech_debt_hours FLOAT,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS developer_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INT REFERENCES predictions(id),
    feedback_type VARCHAR(30),
    note TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ci_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    commit_sha VARCHAR(64),
    gate_decision VARCHAR(10) CHECK (gate_decision IN ('pass','block','overridden')),
    override_reason TEXT,
    deployed_url TEXT,
    run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
