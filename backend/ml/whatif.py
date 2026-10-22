"""
What-If Risk Simulator.

Re-runs inference on user-modified metrics without retraining.
Supports both TCA+-projected and plain-scaled models.
"""
import numpy as np
from typing import Optional

from backend.ml.train import assign_risk_tier, load_artefact


def whatif_predict(
    model_path: str,
    modified_metrics: dict,
    feature_names: Optional[list] = None,
) -> dict:
    """
    Re-run inference on a modified feature vector.

    Parameters
    ----------
    model_path : str
        Path to the pickled model artefact (from train.py).
    modified_metrics : dict
        {feature_name: value} for the modified file metrics.
        Any feature not provided will be set to 0.
    feature_names : list, optional
        Override the feature list (defaults to artefact's feature_names).

    Returns
    -------
    dict with defect_probability, risk_tier, and input features used.
    """
    artefact = load_artefact(model_path)
    clf = artefact["clf"]
    feat_names = feature_names or artefact["feature_names"]
    med_thresh = artefact["medium_threshold"]
    high_thresh = artefact["high_threshold"]

    # Build feature vector
    x = np.array([modified_metrics.get(f, 0.0) for f in feat_names], dtype=float)
    x2d = x.reshape(1, -1)

    if artefact.get("use_transfer") and "tca" in artefact:
        # Project using the fitted TCA instance
        tca = artefact["tca"]
        z = tca.transform_single(x2d)
    elif "scaler" in artefact:
        z = artefact["scaler"].transform(x2d)
    else:
        z = x2d

    prob = float(clf.predict_proba(z)[0, 1])
    tier = assign_risk_tier(prob, med_thresh, high_thresh)

    return {
        "defect_probability": round(prob, 4),
        "risk_tier": tier,
        "input_features": {f: modified_metrics.get(f, 0.0) for f in feat_names},
    }
