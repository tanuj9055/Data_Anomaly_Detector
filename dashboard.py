import pandas as pd
import streamlit as st
from config import OUTPUT_DIR

st.set_page_config(page_title="Data Quality Anomaly Dashboard", layout="wide")

st.title("📊 Data Quality Anomaly Dashboard")
st.markdown(
    "This dashboard visualizes outputs from the anomaly detector on claims data."
)


def load_csv(path):
    try:
        return pd.read_csv(path)
    except:
        return None


summary_path = OUTPUT_DIR / "summary.csv"
sev_path = OUTPUT_DIR / "severity_report.csv"
processed_path = OUTPUT_DIR / "claims_processed.csv"

summary_df = load_csv(summary_path)
sev_df = load_csv(sev_path)
processed_df = load_csv(processed_path)

st.sidebar.header("⚙️ Controls")

if summary_df is not None and not summary_df.empty:
    issues = summary_df["issue_type"].tolist()
    selected_issues = st.sidebar.multiselect(
        "Issue types", options=issues, default=issues
    )
else:
    selected_issues = []

max_rows = st.sidebar.slider("Max rows", 50, 500, 200, 50)

st.sidebar.markdown("---")
st.sidebar.markdown("Use filters to explore specific issues.")

if summary_df is not None and selected_issues:
    summary_filtered = summary_df[summary_df["issue_type"].isin(selected_issues)]
else:
    summary_filtered = summary_df

if sev_df is not None and selected_issues:
    sev_filtered = sev_df[sev_df["issue_type"].isin(selected_issues)]
else:
    sev_filtered = sev_df

st.markdown("## 🔍 Overview")

col1, col2, col3, col4 = st.columns(4)

total_rows = len(processed_df) if processed_df is not None else 0

if summary_filtered is not None and not summary_filtered.empty:
    total_issues = int(summary_filtered["row_count"].sum())
else:
    total_issues = 0

if sev_filtered is not None and not sev_filtered.empty:
    hc = sev_filtered[sev_filtered["severity_label"].isin(["High", "Critical"])]
    high_crit = hc["row_count"].sum()
else:
    high_crit = 0

if sev_filtered is not None and not sev_filtered.empty and total_rows > 0:
    max_score = total_rows * 4
    total_score = sev_filtered["total_severity_score"].sum()
    ratio = min(total_score / max_score, 1)
    dqs = round(100 - ratio * 100, 1)
else:
    dqs = 100.0

col1.metric("Total Rows", total_rows)
col2.metric("Total Issues", total_issues)
col3.metric("High/Critical", high_crit)
col4.metric("Data Quality Score", f"{dqs} / 100")

st.markdown("---")
st.subheader("📌 Summary (Filtered)")

if summary_filtered is not None and not summary_filtered.empty:
    st.dataframe(summary_filtered, use_container_width=True)
    st.bar_chart(summary_filtered.set_index("issue_type")["row_count"])
else:
    st.info("No summary available.")

st.markdown("---")
st.subheader("⚠️ Severity (Filtered)")

if sev_filtered is not None and not sev_filtered.empty:
    with st.expander("Show severity", expanded=True):
        st.dataframe(sev_filtered, use_container_width=True)

    st.bar_chart(sev_filtered.set_index("issue_type")["total_severity_score"])
else:
    st.info("No severity report available.")

st.markdown("---")
st.subheader("🧾 Detailed Issues")

tabs = st.tabs(
    [
        "Duplicates",
        "Missing Values",
        "Invalid Formats",
        "Outliers (IQR)",
        "ML Anomalies",
    ]
)

file_map = {
    "Duplicates": "duplicates.csv",
    "Missing Values": "missing_values.csv",
    "Invalid Formats": "invalid_formats.csv",
    "Outliers (IQR)": "outliers_iqr.csv",
    "ML Anomalies": "anomalies_isolation_forest.csv",
}

for tab, (label, fname) in zip(tabs, file_map.items()):
    with tab:
        path = OUTPUT_DIR / fname
        df = load_csv(path)
        if df is not None and not df.empty:
            st.write(f"{len(df)} rows found in {label}")
            st.dataframe(df.head(max_rows), use_container_width=True)
        else:
            st.warning(f"{fname} missing or empty.")
