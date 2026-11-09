from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.platform import (
    FEATURES,
    audit_log,
    export_report,
    models,
    monitoring_snapshot,
    predict_batch,
    prediction_history,
    production_version,
    set_status,
    simulate_events,
    train_model,
    update_settings,
)

st.set_page_config(
    page_title="Model Operations & Monitoring",
    page_icon="📈",
    layout="wide",
)

# ---------- Visual styling ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1220 0%, #111c31 55%, #102927 100%);
        color: #f3f6fc;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 18, 32, 0.96);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #121e35 0%, #101827 100%);
        border-right: 1px solid #293853;
    }

    h1, h2, h3, h4, p, label, span, li, [data-testid="stCaptionContainer"] {
        color: #e8eef8;
    }

    [data-testid="stMarkdownContainer"] p {
        color: #d3deed;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #17243b, #1b2b46);
        border: 1px solid #344968;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
    }

    [data-testid="stMetricValue"] {
        color: #75a7ff;
        font-weight: 700;
    }

    .stButton > button {
        border-radius: 10px;
        border: 0;
        background: linear-gradient(90deg, #4f46e5, #0ea5a4);
        color: white;
        font-weight: 600;
    }

    .stButton > button:hover {
        border: 0;
        color: white;
        filter: brightness(1.08);
    }

    div[data-testid="stTabs"] button[role="tab"] {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.title("📈 Model Operations & Monitoring")
st.caption(
    "Local-first MLOps portfolio demo · synthetic data · "
    "CPU-friendly · no cloud account"
)

# ---------- Sidebar controls ----------
with st.sidebar:
    st.title("🎛️ Controls")
    st.write("Local MLOps")

    if st.button("🔄 Refresh dashboard", use_container_width=True):
        st.rerun()

    st.divider()
    st.subheader("Demo traffic")

    demo_rows = st.slider(
        "Events to simulate",
        min_value=10,
        max_value=500,
        value=120,
        step=10,
    )
    inject_drift = st.toggle(
        "Add feature drift",
        value=True,
        help="Shifts two synthetic input features to demonstrate data drift.",
    )

    st.divider()
    st.caption(
        "The traffic generator uses synthetic data. "
        "It does not contact an external service."
    )

# ---------- Model setup ----------
versions = models()

if versions.empty:
    st.info("No model versions yet. Train a candidate to initialize the platform.")

    if st.button("🚀 Train first candidate", type="primary"):
        with st.spinner("Training the lightweight model..."):
            result = train_model()

        st.success(
            f"Created {result['version']} · "
            f"F1 score: {result['metrics']['f1']:.3f}"
        )
        st.info(
            "The new version is a candidate. "
            "Use the Model registry tab to validate and promote it."
        )
        st.rerun()

    st.stop()

# ---------- Monitoring summary ----------
snapshot = monitoring_snapshot()
history = prediction_history()

st.subheader("Live monitoring overview")

metric_columns = st.columns(5)
metric_values = [
    ("📦 Predictions", str(snapshot["volume"])),
    ("⚠️ Error rate", f"{snapshot['error_rate']:.1%}"),
    ("⏱️ p95 latency", f"{snapshot['p95_latency_ms']:.2f} ms"),
    (
        "🎯 Mean confidence",
        "—"
        if snapshot["mean_confidence"] is None
        else f"{snapshot['mean_confidence']:.1%}",
    ),
    (
        "🌊 Drifted features",
        f"{len(snapshot['drift']['drifted_features'])}/{len(FEATURES)}",
    ),
]

for column, (label, value) in zip(metric_columns, metric_values):
    column.metric(label, value)

# ---------- Alerts and status ----------
if snapshot["alerts"]:
    st.subheader("🚨 Alerts")
    for alert in snapshot["alerts"]:
        st.warning(alert)
else:
    st.success("✅ Monitoring thresholds are currently clear.")

if snapshot["retraining_recommendation"]:
    st.info(
        "A retraining review is recommended. "
        "This is advisory only: the app will not retrain or promote a model automatically."
    )

if snapshot["labeled_count"] == 0:
    st.caption(
        "Performance metrics are unavailable until ground-truth labels are attached. "
        "Feature drift is an input-distribution signal; it does not prove concept drift "
        "or reduced model accuracy."
    )

# ---------- Main tabs ----------
tab_monitor, tab_infer, tab_registry, tab_audit = st.tabs(
    [
        "📊 Monitor",
        "🧪 Batch scoring",
        "🗂️ Model registry",
        "📝 Audit & reports",
    ]
)

# =====================================================================
# Monitor tab
# =====================================================================
with tab_monitor:
    st.subheader("Prediction monitoring")

    if history.empty:
        st.info(
            "There are no recorded predictions yet. "
            "Open the Batch scoring tab to generate sample events."
        )
    else:
        valid_history = history[history["error"].isna()].copy()

        if not valid_history.empty:
            minimum_confidence = st.slider(
                "Minimum prediction confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                help="Filter the class distribution chart by prediction confidence.",
            )

            visible = valid_history[
                valid_history["confidence"] >= minimum_confidence
            ]

            st.caption(
                f"Showing {len(visible)} of {len(valid_history)} successful "
                "predictions at this confidence setting."
            )

            chart_left, chart_right = st.columns(2)

            with chart_left:
                st.markdown("#### 🎨 Predicted class mix")

                if not visible.empty:
                    distribution = (
                        visible["predicted_class"]
                        .value_counts(normalize=True)
                        .rename_axis("Class")
                        .reset_index(name="Share")
                    )
                    distribution["Class"] = (
                        "Class " + distribution["Class"].astype(str)
                    )

                    class_chart = px.pie(
                        distribution,
                        names="Class",
                        values="Share",
                        hole=0.55,
                        color="Class",
                        color_discrete_sequence=[
                            "#6366f1",
                            "#14b8a6",
                            "#f59e0b",
                            "#ec4899",
                        ],
                        title="Predicted class distribution",
                    )
                    class_chart.update_traces(
                        textposition="inside",
                        textinfo="percent+label",
                        hovertemplate=(
                            "%{label}<br>Share: %{percent}<extra></extra>"
                        ),
                    )
                    class_chart.update_layout(
                        legend_title_text="Predicted class",
                        margin=dict(t=55, b=20, l=20, r=20),
                    )
                    st.plotly_chart(
                        class_chart,
                        use_container_width=True,
                    )
                else:
                    st.info(
                        "No predictions match the selected confidence filter."
                    )

            with chart_right:
                st.markdown("#### 🌊 Feature drift · PSI")

                psi_values = snapshot["drift"]["feature_psi"]
                if psi_values:
                    psi_data = pd.DataFrame(
                        [
                            {
                                "Feature": feature,
                                "PSI": value,
                                "Status": (
                                    "Review"
                                    if value >= 0.1
                                    else "Within screening threshold"
                                ),
                            }
                            for feature, value in psi_values.items()
                        ]
                    )

                    psi_chart = px.bar(
                        psi_data,
                        x="Feature",
                        y="PSI",
                        color="Status",
                        color_discrete_map={
                            "Review": "#f97316",
                            "Within screening threshold": "#14b8a6",
                        },
                        title="Population Stability Index by feature",
                        hover_data={"PSI": ":.3f"},
                    )
                    psi_chart.add_hline(
                        y=0.1,
                        line_dash="dash",
                        line_color="#dc2626",
                        annotation_text="Screening threshold: 0.10",
                    )
                    psi_chart.update_layout(
                        xaxis_title="Input feature",
                        yaxis_title="PSI",
                        legend_title_text="Screening status",
                        margin=dict(t=55, b=20, l=20, r=20),
                    )
                    st.plotly_chart(
                        psi_chart,
                        use_container_width=True,
                    )
                    st.caption(
                        "PSI ≥ 0.10 is used here as a practical screening "
                        "threshold. Investigate the context before acting."
                    )
                else:
                    st.info("No feature drift data is available yet.")

            st.markdown("#### 🔎 Recent predictions")

            display_columns = [
                column
                for column in [
                    "created_at",
                    "model_version",
                    "predicted_class",
                    "confidence",
                    "latency_ms",
                    "true_label",
                ]
                if column in valid_history.columns
            ]

            recent = valid_history[display_columns].tail(20).iloc[::-1]
            st.dataframe(
                recent,
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.subheader("🎚️ Alert thresholds")
    settings = snapshot["settings"]

    with st.form("thresholds"):
        threshold_columns = st.columns(5)

        new_settings = {
            "min_volume": threshold_columns[0].number_input(
                "Minimum volume",
                min_value=0,
                max_value=1_000_000,
                value=int(settings["min_volume"]),
            ),
            "max_error_rate": threshold_columns[1].number_input(
                "Maximum error rate",
                min_value=0.0,
                max_value=1.0,
                value=float(settings["max_error_rate"]),
                step=0.01,
            ),
            "max_p95_latency_ms": threshold_columns[2].number_input(
                "Maximum p95 latency (ms)",
                min_value=0.0,
                max_value=100_000.0,
                value=float(settings["max_p95_latency_ms"]),
            ),
            "max_drift_share": threshold_columns[3].number_input(
                "Maximum drifted feature share",
                min_value=0.0,
                max_value=1.0,
                value=float(settings["max_drift_share"]),
                step=0.05,
            ),
            "min_confidence": threshold_columns[4].number_input(
                "Minimum mean confidence",
                min_value=0.0,
                max_value=1.0,
                value=float(settings["min_confidence"]),
                step=0.01,
            ),
        }

        if st.form_submit_button("💾 Save thresholds"):
            update_settings(new_settings)
            st.success("Thresholds saved.")
            st.rerun()

# =====================================================================
# Batch scoring tab
# =====================================================================
with tab_infer:
    st.subheader("🚦 Simulated production traffic")
    st.write(
        "Generate synthetic incoming records and score them with the current "
        "production model."
    )

    if st.button(
        f"🚀 Generate {demo_rows} sample events",
        type="primary",
    ):
        try:
            with st.spinner("Scoring synthetic events..."):
                simulated = simulate_events(
                    demo_rows,
                    drift=inject_drift,
                )

            st.success(
                f"Recorded {len(simulated)} predictions using "
                f"{production_version() or 'the selected model'}."
            )
            st.dataframe(
                simulated.head(15),
                use_container_width=True,
                hide_index=True,
            )
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("📄 Score a CSV batch")
    st.caption(
        "Your CSV must contain feature_0 through feature_5. "
        "An optional target column is treated as ground truth for this demo."
    )

    uploaded_file = st.file_uploader(
        "Choose a feature CSV",
        type=["csv"],
        help="The CSV must contain feature_0 through feature_5.",
    )

    available_versions = versions["version"].tolist()
    current_production = production_version()

    if current_production in available_versions:
        default_index = available_versions.index(current_production)
    else:
        default_index = 0

    selected_model = st.selectbox(
        "Model version for scoring",
        available_versions,
        index=default_index,
    )

    if uploaded_file and st.button("🧠 Score uploaded batch"):
        try:
            batch = pd.read_csv(uploaded_file)
            scored = predict_batch(batch, selected_model)

            st.success(f"Scored {len(scored)} rows.")
            st.dataframe(
                scored,
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "⬇️ Download predictions CSV",
                data=scored.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Could not score this file: {exc}")

    # Delayed-label workflow
    current_history = prediction_history()

    if not current_history.empty:
        unlabeled = current_history[
            current_history["error"].isna()
            & current_history["true_label"].isna()
        ]

        if not unlabeled.empty:
            with st.expander(
                f"🏷️ Attach labels to {len(unlabeled)} predictions"
            ):
                label_template = unlabeled[
                    ["id", "predicted_class"]
                ].rename(
                    columns={
                        "id": "prediction_id",
                        "predicted_class": "example_prediction",
                    }
                )

                st.download_button(
                    "⬇️ Download label template",
                    data=label_template.to_csv(index=False),
                    file_name="label_template.csv",
                    mime="text/csv",
                )

                st.caption(
                    "Fill in a `target` column with the actual class "
                    "label (0 or 1), then upload the CSV."
                )

                label_file = st.file_uploader(
                    "Upload labeled CSV",
                    type=["csv"],
                    key="label_upload",
                )

                if label_file and st.button("✅ Attach ground truth"):
                    try:
                        from src.platform import attach_labels

                        matched = attach_labels(pd.read_csv(label_file))
                        st.success(f"Attached {matched} labels.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not attach labels: {exc}")

# =====================================================================
# Model registry tab
# =====================================================================
with tab_registry:
    st.subheader("🗂️ Model versions and lifecycle")

    registry_view = versions.copy()
    registry_view["metrics"] = registry_view["metrics"].map(
        lambda value: json.dumps(
            json.loads(value),
            sort_keys=True,
        )
    )

    st.dataframe(
        registry_view[
            [
                "version",
                "created_at",
                "status",
                "metrics",
                "seed",
                "run_id",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Change a model's lifecycle status")

    selected_version = st.selectbox(
        "Choose a model version",
        versions["version"].tolist(),
        key="registry_version",
    )

    current_status = versions.loc[
        versions["version"] == selected_version,
        "status",
    ].iloc[0]

    lifecycle_statuses = [
        "candidate",
        "validated",
        "production",
        "archived",
    ]

    chosen_status = st.selectbox(
        "New status",
        lifecycle_statuses,
        index=lifecycle_statuses.index(current_status),
    )

    if st.button("🔁 Apply lifecycle change"):
        try:
            set_status(selected_version, chosen_status)
            st.success(
                f"{selected_version} is now marked as {chosen_status}."
            )
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("🧪 Train another candidate")

    training_seed = st.number_input(
        "Training seed",
        min_value=0,
        max_value=1_000_000,
        value=41,
        step=1,
    )

    if st.button("🌱 Train new candidate"):
        try:
            with st.spinner("Training a new model version..."):
                result = train_model(seed=int(training_seed))

            st.success(
                f"Created {result['version']} · "
                f"F1 score: {result['metrics']['f1']:.3f}"
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    if len(versions) >= 2:
        st.divider()
        st.subheader("⚖️ Compare model versions")

        default_versions = versions["version"].tolist()[:2]
        versions_to_compare = st.multiselect(
            "Select versions to compare",
            options=versions["version"].tolist(),
            default=default_versions,
        )

        comparison_rows = []

        for _, model_row in versions[
            versions["version"].isin(versions_to_compare)
        ].iterrows():
            metrics = json.loads(model_row["metrics"])
            comparison_rows.append(
                {
                    "Version": model_row["version"],
                    "Status": model_row["status"],
                    **metrics,
                }
            )

        if comparison_rows:
            comparison = pd.DataFrame(comparison_rows)
            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True,
            )

            metric_columns = [
                column
                for column in ["accuracy", "precision", "recall", "f1"]
                if column in comparison.columns
            ]

            if metric_columns:
                long_comparison = comparison.melt(
                    id_vars=["Version"],
                    value_vars=metric_columns,
                    var_name="Metric",
                    value_name="Score",
                )

                comparison_chart = px.bar(
                    long_comparison,
                    x="Metric",
                    y="Score",
                    color="Version",
                    barmode="group",
                    range_y=[0, 1],
                    color_discrete_sequence=[
                        "#6366f1",
                        "#14b8a6",
                        "#f59e0b",
                        "#ec4899",
                    ],
                    title="Holdout metric comparison",
                )
                st.plotly_chart(
                    comparison_chart,
                    use_container_width=True,
                )

# =====================================================================
# Audit and reports tab
# =====================================================================
with tab_audit:
    st.subheader("📝 Audit events")

    event_log = audit_log()
    if event_log.empty:
        st.info("No audit events recorded yet.")
    else:
        st.dataframe(
            event_log,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("📤 Export")

    st.download_button(
        "⬇️ Export monitoring report (JSON)",
        data=export_report(),
        file_name="monitoring-report.json",
        mime="application/json",
    )

    export_history = prediction_history()
    if not export_history.empty:
        st.download_button(
            "⬇️ Export prediction history (CSV)",
            data=export_history.to_csv(index=False),
            file_name="prediction-history.csv",
            mime="text/csv",
        )
