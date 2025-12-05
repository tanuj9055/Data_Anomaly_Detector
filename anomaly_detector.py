import logging
from datetime import datetime
import pandas as pd
from sklearn.ensemble import IsolationForest
from config import IFOREST_CONTAMINATION, IFOREST_ESTIMATORS, RANDOM_SEED

logger = logging.getLogger(__name__)


def _bad_dt(v):
    try:
        datetime.strptime(str(v), "%Y-%m-%d")
        return False
    except:
        return True


class DataQualityAnomalyDetector:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results = {}

        self.df["missing_count"] = self.df.isna().sum(1)

        z = self.df["zip_code"].astype(str)
        self.df["invalid_zip_flag"] = (~z.str.fullmatch(r"\d{5}", na=False)).astype(int)

        self.df["invalid_dob_flag"] = self.df["dob"].apply(_bad_dt).astype(int)

        self.df["duplicate_flag"] = self.df["claim_id"].duplicated(keep=False).astype(int)

    def find_duplicates(self, subset=None):
        sub = subset if subset else ["claim_id"]
        d = self.df[self.df.duplicated(subset=sub, keep=False)]
        self.results["duplicates"] = d
        return d

    def find_missing_values(self):
        m = self.df[self.df.isna().any(axis=1)]
        self.results["missing_values"] = m
        return m

    def find_invalid_formats(self):
        df = self.df
        z = ~df["zip_code"].astype(str).str.fullmatch(r"\d{5}", na=False)
        d = df["dob"].apply(_bad_dt)
        out = df[z | d]
        self.results["invalid_formats"] = out
        return out

    def find_outliers_iqr(self, column="claim_amount"):
        s = self.df[column]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out = self.df[(s < low) | (s > high)]
        self.results["outliers_iqr"] = out
        return out

    def find_anomalies_isolation_forest(self, feature_cols=None, contamination=IFOREST_CONTAMINATION):
        cols = feature_cols or ["claim_amount", "doctor_id", "patient_id"]
        tmp = self.df[cols].dropna()

        if tmp.empty:
            self.results["anomalies_isolation_forest"] = pd.DataFrame()
            return pd.DataFrame()

        mdl = IsolationForest(
            n_estimators=IFOREST_ESTIMATORS,
            contamination=contamination,
            random_state=RANDOM_SEED,
        )
        mdl.fit(tmp)

        p = mdl.predict(tmp)
        idx = tmp.index[p == -1]

        out = self.df.loc[idx]
        self.results["anomalies_isolation_forest"] = out
        return out

    def summary(self):
        s = []
        for k, v in self.results.items():
            s.append({"issue_type": k, "row_count": len(v)})
        return pd.DataFrame(s)

    def severity_report(self):
        sev = {
            "duplicates": ("High", 3),
            "missing_values": ("Medium", 2),
            "invalid_formats": ("High", 3),
            "outliers_iqr": ("Medium", 2),
            "anomalies_isolation_forest": ("Critical", 4),
        }

        sm = self.summary()
        if sm.empty:
            return sm

        out = []
        for _, r in sm.iterrows():
            label, sc = sev.get(r["issue_type"], ("Low", 1))
            out.append(
                {
                    "issue_type": r["issue_type"],
                    "row_count": r["row_count"],
                    "severity_label": label,
                    "severity_score": sc,
                    "total_severity_score": sc * r["row_count"],
                }
            )
        return pd.DataFrame(out)
