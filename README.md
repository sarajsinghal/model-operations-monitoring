# Model Operations & Monitoring

A runnable, local-first portfolio project showing an end-to-end lightweight ML lifecycle: deterministic training, versioned model artifacts, a governed registry, batch scoring, simulated production traffic, operational and data-drift monitoring, optional labels, threshold alerts, an audit trail, and report export. It is designed to run on Windows 11 with an 8 GB RAM laptop CPU; no GPU, Docker, credentials, cloud service, or internet connection is needed after dependencies are installed.

**Live app:** https://model-operations-monitoring.streamlit.app/

## What is implemented

- **Training:** reproducible scikit-learn `StandardScaler` + `LogisticRegression` pipeline on deterministic synthetic binary classification data. `--seed` makes the data split and fit reproducible. Each run gets a unique version and a saved `joblib` artifact.
- **Tracking:** local SQLite stores model metadata, metrics, artifact path, registry state, prediction records, settings, and audit events. MLflow is optional; installing `requirements-mlflow.txt` logs local experiment parameters and metrics into `mlruns/`. The core app works without it.
- **Lifecycle:** `candidate`, `validated`, `production`, and `archived` states. Promotion is always a human action; promoting one version archives the former production version.
- **Scoring:** CSV batch scoring through the UI or Python API. Prediction class, confidence, latency, model version, features, and errors are recorded. No network API server is included; the dashboard is the inference interface.
- **Monitoring:** prediction volume, error rate, p95 latency, predicted class mix, mean confidence, per-feature PSI, configurable thresholds, and advisory retraining recommendations. Monitoring does not retrain or promote models.
- **Delayed labels:** a downloadable label template accepts `prediction_id,target`; accuracy, precision, recall and F1 become visible after labels are matched to scored events.
- **Reports:** downloadable JSON monitoring report and CSV prediction history.
- **Demo data:** generated deterministically in memory when needed. No sample input file needs to be committed or deployed.

## Data drift and concept drift

**Data drift** means the observed input feature distribution changed. This demo screens numeric features with Population Stability Index (PSI), using deciles from the training reference distribution and flags PSI ≥ 0.10. This is a practical heuristic, not a statistical significance test or proof that a model is harmed.

**Concept drift** means the relationship between inputs and the target changed. Input-only monitoring cannot confirm it. A shifted class distribution or low confidence is not proof of concept drift, and data drift alone does not prove accuracy has fallen. Performance degradation can be evaluated only after representative ground-truth labels arrive. Even then, label delay, selection bias, and sample size affect the conclusion. The dashboard keeps these signals distinct.

## Windows PowerShell setup

Open PowerShell in this folder:

```powershell
cd "$HOME\Documents\Codex\2026-11-08\referenced-chatgpt-conversation-this-is-an-12\project14_mlops"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can skip activation and call the virtual environment directly: `\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `\.venv\Scripts\streamlit.exe run app.py`. Python 3.11 is recommended. After installation, the app and scripts run offline.

## Run the project

Train a first version, then explicitly validate and promote it:

```powershell
python train.py --seed 41
python manage.py list
python manage.py status <VERSION> validated
python manage.py status <VERSION> production
```

Start the dashboard:

```powershell
streamlit run app.py
```

The dashboard can train the first candidate itself. Use **Batch scoring** to generate 120 simulated events; shifted features are enabled by default to demonstrate input drift. Use **Model registry** to compare runs and change lifecycle status. Use **Audit & reports** to export results.

To score a CSV you provide from PowerShell after a production version exists:

```powershell
python predict.py C:\\path\\to\\your\\input.csv --output predictions.csv
```

Optional MLflow:

```powershell
pip install -r requirements-mlflow.txt
streamlit run app.py
```

Then use MLflow's local UI if desired:

```powershell
mlflow ui --backend-store-uri .\mlruns --host 127.0.0.1
```

## Project structure

```text
project14_mlops/
├── app.py                    # Streamlit monitoring and operations dashboard
├── train.py / predict.py     # Command-line workflows
├── manage.py                 # Registry, simulation, labels, report commands
├── src/platform.py           # Training, persistence, inference and monitoring logic
├── tests/                    # Core workflow and drift tests
├── requirements.txt          # Lightweight local stack
└── requirements-mlflow.txt   # Optional local experiment tracker
```

Runtime files are created under `data/` and `artifacts/` and ignored by Git. Do not commit uploaded CSVs or generated reports. Treat local runtime files as sensitive if you later score real data. SQLite is not encrypted and local artifacts are not access-controlled.

## Tests

```powershell
pytest -q
```

The tests use temporary directories and verify training, model lifecycle, batch inference, event persistence, optional ground truth, drift behavior, and report output.

## Honest scope and limitations

This is a local portfolio demonstration, not a hardened production service. The traffic generator is simulated; latency is measured locally; SQLite is single-machine storage; there is no authentication, network inference API, scheduler, notification transport, automatic retraining, automatic promotion, or cloud deployment. MLflow is optional and local. Drift thresholds are illustrative and should be calibrated to a real domain. Model registry status is an auditable SQLite lifecycle, not a vendor registry service.

## Portfolio presentation

Demo sequence: train a candidate, show its holdout metrics and saved version, validate and promote it with explicit clicks, score a normal sample and then simulate shifted traffic, explain the PSI flags and the difference between data and concept drift, attach labels to show delayed performance metrics, compare model versions, and export the monitoring report. Mention that operational metrics and input drift are immediately available while performance claims require labels.

Truthful resume bullets:

- Built a local-first MLOps monitoring dashboard with reproducible scikit-learn training, versioned artifacts, an SQLite-backed model lifecycle, and batch prediction.
- Implemented prediction volume, error, latency, confidence, class-mix, and PSI-based feature-drift monitoring with configurable thresholds and human-reviewed retraining recommendations.
- Added delayed-label evaluation, audit events, model version comparison, exportable reports, deterministic synthetic demo data, and optional local MLflow experiment logging.
