from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [f"feature_{i}" for i in range(6)]
TARGET = "target"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def generate_dataset(n: int = 1200, seed: int = 41) -> pd.DataFrame:
    """Deterministic, fictional numeric data; it contains no personal information."""
    x, y = make_classification(n_samples=n, n_features=6, n_informative=4,
                               n_redundant=1, n_clusters_per_class=1,
                               class_sep=1.15, weights=[0.55, 0.45],
                               random_state=seed)
    frame = pd.DataFrame(x, columns=FEATURES)
    frame[TARGET] = y.astype(int)
    return frame


def initialize(root: Path = ROOT) -> None:
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    db = root / "data" / "platform.sqlite3"
    with sqlite3.connect(db) as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS models (
            version TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL,
            metrics TEXT NOT NULL, artifact_path TEXT NOT NULL, data_hash TEXT NOT NULL,
            seed INTEGER NOT NULL, run_id TEXT
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, model_version TEXT NOT NULL,
            latency_ms REAL NOT NULL, error TEXT, predicted_class INTEGER,
            confidence REAL, features TEXT NOT NULL, true_label INTEGER
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            action TEXT NOT NULL, details TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value REAL NOT NULL);
        """)
        defaults = {"min_volume": 5, "max_error_rate": 0.05, "max_p95_latency_ms": 250.0,
                    "max_drift_share": 0.25, "min_confidence": 0.55}
        for key, value in defaults.items():
            con.execute("INSERT OR IGNORE INTO settings VALUES (?, ?)", (key, value))


def _event(con: sqlite3.Connection, action: str, details: dict[str, Any]) -> None:
    con.execute("INSERT INTO events(created_at, action, details) VALUES (?, ?, ?)",
                (now(), action, json.dumps(details, sort_keys=True)))


def _mlflow_log(version: str, metrics: dict[str, float], seed: int) -> str | None:
    try:
        import mlflow
        mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
        mlflow.set_experiment("project14-local")
        with mlflow.start_run(run_name=version) as run:
            mlflow.log_params({"seed": seed, "estimator": "LogisticRegression", "features": len(FEATURES)})
            mlflow.log_metrics(metrics)
            return run.info.run_id
    except ImportError:
        return None


def train_model(seed: int = 41, root: Path = ROOT) -> dict[str, Any]:
    initialize(root)
    frame = generate_dataset(seed=seed)
    train, test = train_test_split(frame, test_size=0.25, random_state=seed, stratify=frame[TARGET])
    model = Pipeline([("scale", StandardScaler()),
                      ("classifier", LogisticRegression(max_iter=500, random_state=seed))])
    model.fit(train[FEATURES], train[TARGET])
    pred = model.predict(test[FEATURES])
    metrics = {"accuracy": float(accuracy_score(test[TARGET], pred)),
               "precision": float(precision_score(test[TARGET], pred, zero_division=0)),
               "recall": float(recall_score(test[TARGET], pred, zero_division=0)),
               "f1": float(f1_score(test[TARGET], pred, zero_division=0))}
    # Content identity is stable for a seed and dataset; distinct training runs still have unique versions.
    version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
    artifact = root / "artifacts" / f"{version}.joblib"
    joblib.dump(model, artifact)
    digest = str(abs(hash(frame.to_csv(index=False))))
    run_id = _mlflow_log(version, metrics, seed)
    with sqlite3.connect(root / "data" / "platform.sqlite3") as con:
        con.execute("INSERT INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (version, now(), "candidate", json.dumps(metrics), str(artifact), digest, seed, run_id))
        _event(con, "model_trained", {"version": version, "metrics": metrics, "seed": seed,
                                       "tracker": "mlflow" if run_id else "sqlite_only"})
    return {"version": version, "metrics": metrics, "artifact_path": str(artifact), "run_id": run_id}


def models(root: Path = ROOT) -> pd.DataFrame:
    initialize(root)
    with sqlite3.connect(root / "data" / "platform.sqlite3") as con:
        rows = con.execute("SELECT version, created_at, status, metrics, artifact_path, seed, run_id FROM models ORDER BY created_at DESC").fetchall()
    return pd.DataFrame(rows, columns=["version", "created_at", "status", "metrics", "artifact_path", "seed", "run_id"])


def set_status(version: str, status: str, root: Path = ROOT) -> None:
    allowed = {"candidate", "validated", "production", "archived"}
    if status not in allowed:
        raise ValueError(f"Unsupported status: {status}")
    initialize(root)
    with sqlite3.connect(root / "data" / "platform.sqlite3") as con:
        found = con.execute("SELECT 1 FROM models WHERE version=?", (version,)).fetchone()
        if not found:
            raise ValueError(f"Unknown model version: {version}")
        if status == "production":
            con.execute("UPDATE models SET status='archived' WHERE status='production'")
        con.execute("UPDATE models SET status=? WHERE version=?", (status, version))
        _event(con, "model_status_changed", {"version": version, "status": status})


def production_version(root: Path = ROOT) -> str | None:
    initialize(root)
    with sqlite3.connect(root / "data" / "platform.sqlite3") as con:
        row = con.execute("SELECT version FROM models WHERE status='production' ORDER BY created_at DESC LIMIT 1").fetchone()
    return row[0] if row else None


def _model(version: str, root: Path) -> Any:
    with sqlite3.connect(root / "data" / "platform.sqlite3") as con:
        row = con.execute("SELECT artifact_path FROM models WHERE version=?", (version,)).fetchone()
    if not row:
        raise ValueError(f"Unknown model version: {version}")
    return joblib.load(row[0])


def predict_batch(frame: pd.DataFrame, version: str | None = None, root: Path = ROOT) -> pd.DataFrame:
    initialize(root)
    version = version or production_version(root)
    if not version:
        raise ValueError("No production model. Train, validate, and promote a model first.")
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing features: {', '.join(sorted(missing))}")
    estimator = _model(version, root)
    output = []
    with sqlite3.connect(root / "data" / "platform.sqlite3") as con:
        for _, row in frame[FEATURES].iterrows():
            started = time.perf_counter()
            try:
                x = row.to_frame().T
                cls = int(estimator.predict(x)[0])
                probabilities = estimator.predict_proba(x)[0]
                confidence = float(np.max(probabilities))
                latency = (time.perf_counter() - started) * 1000
                pred_id = uuid.uuid4().hex
                features_json = json.dumps({k: float(row[k]) for k in FEATURES})
                con.execute("INSERT INTO predictions(id, created_at, model_version, latency_ms, predicted_class, confidence, features, true_label) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                             (pred_id, now(), version, latency, cls, confidence, features_json,
                              int(row[TARGET]) if TARGET in frame.columns and pd.notna(row.get(TARGET)) else None))
                output.append({"prediction_id": pred_id, "model_version": version,
                               "predicted_class": cls, "confidence": confidence, "latency_ms": latency,
                               **{k: float(row[k]) for k in FEATURES}})
            except Exception as exc:
                latency = (time.perf_counter() - started) * 1000
                pred_id = uuid.uuid4().hex
                con.execute("INSERT INTO predictions(id, created_at, model_version, latency_ms, error, features) VALUES (?, ?, ?, ?, ?, ?)",
                             (pred_id, now(), version, latency, str(exc), json.dumps({})))
                output.append({"prediction_id": pred_id, "model_version": version, "error": str(exc), "latency_ms": latency})
        _event(con, "batch_scored", {"version": version, "rows": len(frame)})
    return pd.DataFrame(output)


def simulate_events(n: int = 120, drift: bool = True, seed: int = 99, root: Path = ROOT) -> pd.DataFrame:
    """Create deterministic fictional incoming records and persist their predictions."""
    frame = generate_dataset(n, seed)
    if drift:
        frame[FEATURES[0]] += 1.35
        frame[FEATURES[1]] += 0.9
    return predict_batch(frame, root=root)


def attach_labels(labels: pd.DataFrame, root: Path = ROOT) -> int:
    """Match labels by prediction_id; the optional true_label column is never inferred."""
    if not {"prediction_id", TARGET}.issubset(labels.columns):
        raise ValueError("Labels CSV must include prediction_id and target columns.")
    initialize(root)
    count = 0
    with sqlite3.connect(root / "data" / "platform.sqlite3") as con:
        for row in labels[["prediction_id", TARGET]].itertuples(index=False):
            cur = con.execute("UPDATE predictions SET true_label=? WHERE id=? AND error IS NULL",
                              (int(row.target), str(row.prediction_id)))
            count += cur.rowcount
        _event(con, "ground_truth_attached", {"matched_labels": count})
    return count


def prediction_history(root: Path = ROOT) -> pd.DataFrame:
    initialize(root)
    with sqlite3.connect(root / "data" / "platform.sqlite3") as con:
        return pd.read_sql_query("SELECT * FROM predictions ORDER BY created_at", con)


def drift_report(history: pd.DataFrame, baseline: pd.DataFrame | None = None) -> dict[str, Any]:
    """Practical numeric data drift: fraction of features over 0.10 PSI; no labels required."""
    if history.empty:
        return {"feature_psi": {}, "drifted_features": [], "drift_share": 0.0}
    if baseline is None:
        baseline = generate_dataset()
    current = pd.DataFrame([json.loads(v) for v in history["features"] if v and v != "{}"])
    if current.empty:
        return {"feature_psi": {}, "drifted_features": [], "drift_share": 0.0}
    psi_values: dict[str, float] = {}
    for feature in FEATURES:
        ref = baseline[feature].to_numpy()
        cur = current[feature].dropna().to_numpy()
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
        if len(edges) < 3 or cur.size == 0:
            psi_values[feature] = 0.0
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        a = np.histogram(ref, bins=edges)[0] / len(ref)
        b = np.histogram(cur, bins=edges)[0] / len(cur)
        a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
        psi_values[feature] = float(np.sum((b - a) * np.log(b / a)))
    drifted = [feature for feature, value in psi_values.items() if value >= 0.1]
    return {"feature_psi": psi_values, "drifted_features": drifted,
            "drift_share": len(drifted) / len(FEATURES)}


def monitoring_snapshot(root: Path = ROOT) -> dict[str, Any]:
    initialize(root)
    history = prediction_history(root)
    total = len(history)
    good = history[history["error"].isna()] if total else history
    probs = good["confidence"].dropna() if total else pd.Series(dtype=float)
    classes = good["predicted_class"].value_counts(normalize=True).to_dict() if total else {}
    drift = drift_report(history)
    metrics: dict[str, Any] = {"volume": total,
                               "error_rate": float(history["error"].notna().mean()) if total else 0.0,
                               "p95_latency_ms": float(good["latency_ms"].quantile(.95)) if len(good) else 0.0,
                               "mean_confidence": float(probs.mean()) if len(probs) else None,
                               "class_distribution": {int(k): float(v) for k, v in classes.items()},
                               "drift": drift}
    labeled = good[good["true_label"].notna()] if total else good
    if len(labeled):
        actual = labeled["true_label"].astype(int)
        predicted = labeled["predicted_class"].astype(int)
        metrics["accuracy"] = float(accuracy_score(actual, predicted))
        for metric, fn in [("precision", precision_score), ("recall", recall_score),
                           ("f1", f1_score)]:
            metrics[metric] = float(fn(actual, predicted, zero_division=0))
        metrics["labeled_count"] = len(labeled)
    else:
        metrics["labeled_count"] = 0
    with sqlite3.connect(root / "data" / "platform.sqlite3") as con:
        settings = dict(con.execute("SELECT key, value FROM settings").fetchall())
    alerts = []
    if total < settings["min_volume"]: alerts.append(f"Low prediction volume: {total} < {int(settings['min_volume'])}")
    if metrics["error_rate"] > settings["max_error_rate"]: alerts.append("Prediction error rate exceeds threshold")
    if metrics["p95_latency_ms"] > settings["max_p95_latency_ms"]: alerts.append("p95 prediction latency exceeds threshold")
    if drift["drift_share"] > settings["max_drift_share"]: alerts.append("Feature drift share exceeds threshold")
    if metrics["mean_confidence"] is not None and metrics["mean_confidence"] < settings["min_confidence"]:
        alerts.append("Mean prediction confidence is below threshold")
    metrics["alerts"] = alerts
    metrics["settings"] = settings
    # Trigger is advisory only: it never changes model state or retrains automatically.
    metrics["retraining_recommendation"] = bool(alerts or (metrics.get("labeled_count", 0) >= 20 and metrics.get("f1", 1.0) < 0.7))
    return metrics


def update_settings(values: dict[str, float], root: Path = ROOT) -> None:
    allowed = {"min_volume", "max_error_rate", "max_p95_latency_ms", "max_drift_share", "min_confidence"}
    if set(values) - allowed:
        raise ValueError("Unknown monitoring setting")
    initialize(root)
    with sqlite3.connect(root / "data" / "platform.sqlite3") as con:
        for key, value in values.items():
            if value < 0 or (key in {"max_error_rate", "max_drift_share", "min_confidence"} and value > 1):
                raise ValueError(f"Invalid threshold for {key}")
            con.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, float(value)))
        _event(con, "thresholds_updated", values)


def audit_log(root: Path = ROOT) -> pd.DataFrame:
    initialize(root)
    with sqlite3.connect(root / "data" / "platform.sqlite3") as con:
        return pd.read_sql_query("SELECT * FROM events ORDER BY id DESC", con)


def export_report(root: Path = ROOT) -> bytes:
    snapshot = monitoring_snapshot(root)
    report = {"generated_at": now(), "monitoring": snapshot,
              "models": models(root).drop(columns=["artifact_path"]).to_dict(orient="records")}
    return json.dumps(report, indent=2, default=str).encode("utf-8")
