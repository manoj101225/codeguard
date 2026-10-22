"""
SQLAlchemy ORM models for CodeGuard (SQLite backend).
"""
import json
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    Text, DateTime, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(150), nullable=False)
    role = Column(String(20))  # 'source' or 'target'
    repo_url = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    file_metrics = relationship("FileMetrics", back_populates="project")


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_project_id = Column(Integer, ForeignKey("projects.id"))
    target_project_id = Column(Integer, ForeignKey("projects.id"))
    used_transfer_learning = Column(Boolean)
    f1_score = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    p_value = Column(Float)
    trained_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="model_run")


class FileMetrics(Base):
    __tablename__ = "file_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    target_project_id = Column(Integer, ForeignKey("projects.id"))
    file_path = Column(Text, nullable=False)
    loc = Column(Integer)
    cyclomatic_complexity = Column(Integer)
    coupling = Column(Integer)
    num_methods = Column(Integer)
    churn_count = Column(Integer)
    test_coverage_pct = Column(Float)
    extracted_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="file_metrics")
    predictions = relationship("Prediction", back_populates="file_metrics_obj")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_run_id = Column(Integer, ForeignKey("model_runs.id"))
    file_metrics_id = Column(Integer, ForeignKey("file_metrics.id"))
    defect_probability = Column(Float)
    risk_tier = Column(String(10))
    top_factors = Column(Text)  # JSON string
    nlp_explanation = Column(Text)
    recommendation = Column(Text)
    tech_debt_hours = Column(Float)
    predicted_at = Column(DateTime, default=datetime.utcnow)

    model_run = relationship("ModelRun", back_populates="predictions")
    file_metrics_obj = relationship("FileMetrics", back_populates="predictions")
    feedbacks = relationship("DeveloperFeedback", back_populates="prediction")

    def get_top_factors(self):
        if self.top_factors:
            return json.loads(self.top_factors)
        return []

    def set_top_factors(self, factors: list):
        self.top_factors = json.dumps(factors)


class DeveloperFeedback(Base):
    __tablename__ = "developer_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    feedback_type = Column(String(30))  # 'false_positive', 'confirmed', etc.
    note = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    prediction = relationship("Prediction", back_populates="feedbacks")


class CIRun(Base):
    __tablename__ = "ci_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    commit_sha = Column(String(64))
    gate_decision = Column(String(10))  # 'pass', 'block', 'overridden'
    override_reason = Column(Text)
    deployed_url = Column(Text)
    run_at = Column(DateTime, default=datetime.utcnow)


# ---- Database connection helper ----

_engine = None
_SessionLocal = None


def get_engine(db_url: str = "sqlite:///./codeguard.db"):
    global _engine
    if _engine is None:
        _engine = create_engine(db_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(_engine)
    return _engine


def get_session():
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(bind=engine)
    return _SessionLocal()
