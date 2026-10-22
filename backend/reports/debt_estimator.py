"""
Technical debt estimator using a SQALE-inspired formula and project-level metrics.

Formula (per file):
  base_hours = (complexity_excess * 0.5) + (loc_excess / 100 * 1.0) + (coupling_excess * 0.3)

where excess = max(0, actual - threshold).

Thresholds (configurable):
  - complexity: CC > 15 triggers debt
  - loc:        LOC > 200 triggers debt
  - coupling:   coupling > 5 triggers debt
"""


COMPLEXITY_THRESHOLD = 15
LOC_THRESHOLD = 200
COUPLING_THRESHOLD = 5

# Multipliers in hours
COMPLEXITY_HOUR_PER_UNIT  = 0.5
LOC_HOUR_PER_100           = 1.0
COUPLING_HOUR_PER_UNIT     = 0.3

# Risk-tier surcharge (extra hours for higher tiers)
TIER_SURCHARGE = {"Low": 0.0, "Medium": 0.5, "High": 1.5}


def estimate_tech_debt(
    cyclomatic_complexity: int,
    loc: int,
    coupling: int,
    risk_tier: str = "Low",
) -> float:
    """
    Estimate hours-to-fix for a file.

    Returns a float (hours), rounded to 1 decimal.
    """
    cc_excess  = max(0, cyclomatic_complexity - COMPLEXITY_THRESHOLD)
    loc_excess = max(0, loc - LOC_THRESHOLD)
    cpl_excess = max(0, coupling - COUPLING_THRESHOLD)

    hours = (
        cc_excess  * COMPLEXITY_HOUR_PER_UNIT +
        (loc_excess / 100) * LOC_HOUR_PER_100 +
        cpl_excess * COUPLING_HOUR_PER_UNIT +
        TIER_SURCHARGE.get(risk_tier, 0.0)
    )
    return round(hours, 1)


def project_debt_summary(file_debts: list[dict]) -> dict:
    """
    Aggregate tech-debt across a project.

    Parameters
    ----------
    file_debts : list of dicts with keys: file_path, tech_debt_hours, risk_tier

    Returns
    -------
    dict with total_hours, by_tier breakdown, worst_offenders (top 5)
    """
    total = sum(d.get("tech_debt_hours", 0.0) for d in file_debts)
    by_tier = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
    for d in file_debts:
        tier = d.get("risk_tier", "Low")
        by_tier[tier] = by_tier.get(tier, 0.0) + d.get("tech_debt_hours", 0.0)

    worst = sorted(file_debts, key=lambda d: d.get("tech_debt_hours", 0.0), reverse=True)[:5]

    return {
        "total_hours":  round(total, 1),
        "by_tier":      {k: round(v, 1) for k, v in by_tier.items()},
        "worst_offenders": [
            {"file_path": d["file_path"], "hours": d.get("tech_debt_hours", 0.0)}
            for d in worst
        ],
    }


def project_health_score(predictions: list[dict]) -> int:
    """
    Compute a 0-100 project health score.

    Score = 100 - (weighted_defect_sum / n_files * 100)
    Weights: High=1.0, Medium=0.5, Low=0.1
    """
    if not predictions:
        return 100
    weights = {"High": 1.0, "Medium": 0.5, "Low": 0.1}
    n = len(predictions)
    wsum = sum(weights.get(p.get("risk_tier", "Low"), 0.0) for p in predictions)
    raw = 100 - (wsum / n * 100)
    return max(0, min(100, int(round(raw))))


def estimated_post_release_risk(
    predictions: list[dict],
    historical_high_defect_rate: float = 0.65,
) -> float:
    """
    Estimated post-release risk = count(High-risk files) × historical defect rate of High-tier.
    """
    high_count = sum(1 for p in predictions if p.get("risk_tier") == "High")
    return round(high_count * historical_high_defect_rate, 2)


def hotspot_rank(predictions: list[dict], centrality_map: dict) -> list[dict]:
    """
    Rank files by risk_score × dependency_centrality.

    Parameters
    ----------
    predictions : list of prediction dicts (must have defect_probability, file_path, risk_tier)
    centrality_map : {file_path: centrality_score [0..1]}

    Returns
    -------
    Same list, sorted by hotspot_score descending, with hotspot_score added.
    """
    tier_weights = {"High": 1.0, "Medium": 0.6, "Low": 0.2}
    for p in predictions:
        prob = p.get("defect_probability", 0.0)
        tier_w = tier_weights.get(p.get("risk_tier", "Low"), 0.2)
        cent = centrality_map.get(p.get("file_path", ""), 0.0)
        # Centrality boosts score; even a 0-centrality file keeps its base score
        p["hotspot_score"] = round(prob * tier_w * (1 + cent * 5), 4)

    return sorted(predictions, key=lambda p: p.get("hotspot_score", 0.0), reverse=True)
