import json
import pandas as pd
import pytest

from src.platform import (FEATURES, TARGET, attach_labels, audit_log, drift_report,
                          export_report, generate_dataset, models, monitoring_snapshot,
                          predict_batch, production_version, set_status, simulate_events,
                          train_model)


def test_full_lifecycle_batch_events_labels_and_report(tmp_path):
    first = train_model(root=tmp_path)
    assert first["metrics"]["f1"] > 0.5
    assert models(tmp_path).iloc[0]["status"] == "candidate"
    with pytest.raises(ValueError):
        predict_batch(generate_dataset(4), root=tmp_path)
    set_status(first["version"], "validated", tmp_path)
    set_status(first["version"], "production", tmp_path)
    incoming = generate_dataset(12)
    scored = predict_batch(incoming, root=tmp_path)
    assert len(scored) == 12
    assert scored["confidence"].between(0, 1).all()
    assert production_version(tmp_path) == first["version"]
    assert monitoring_snapshot(tmp_path)["labeled_count"] == 0
    labels = scored[["prediction_id"]].copy()
    labels[TARGET] = incoming[TARGET].to_numpy()
    assert attach_labels(labels, tmp_path) == 12
    snapshot = monitoring_snapshot(tmp_path)
    assert snapshot["labeled_count"] == 12
    assert "f1" in snapshot
    assert len(audit_log(tmp_path)) >= 5
    assert json.loads(export_report(tmp_path))["monitoring"]["volume"] == 12


def test_production_promotion_archives_previous(tmp_path):
    one = train_model(root=tmp_path)
    two = train_model(seed=42, root=tmp_path)
    set_status(one["version"], "production", tmp_path)
    set_status(two["version"], "production", tmp_path)
    status = models(tmp_path).set_index("version")["status"]
    assert status[one["version"]] == "archived"
    assert status[two["version"]] == "production"


def test_data_drift_flag_is_not_concept_drift_claim():
    baseline = generate_dataset(500)
    shifted = baseline[FEATURES].copy()
    shifted[FEATURES[0]] += 2.0
    history = pd.DataFrame({"features": [row.to_json() for _, row in shifted.iterrows()]})
    result = drift_report(history, baseline)
    assert FEATURES[0] in result["drifted_features"]
    # The function returns input-distribution evidence only; it has no target or performance claim.
    assert "concept_drift" not in result
    assert "accuracy" not in result
