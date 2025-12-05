import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple
import logging
from config import OUTPUT_DIR, RANDOM_SEED, N_ROWS

logger = logging.getLogger(__name__)


def generate_mock_claims_data(
    n_rows: int = N_ROWS,
    dup_fraction: float = 0.05,
    missing_fraction: float = 0.03,
    invalid_fraction: float = 0.03,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:

    rng = np.random.default_rng(random_seed)

    claim_ids = np.arange(1, n_rows + 1)
    patient_ids = rng.integers(1000, 1000 + n_rows, n_rows)
    doctor_ids = rng.integers(500, 800, n_rows)
    diagnosis_codes = rng.choice(["E11", "I10", "J45", "C34", "K21"], n_rows)
    procedure_codes = rng.choice(["99213", "93000", "81002", "70450", "45378"], n_rows)
    statuses = rng.choice(["APPROVED", "DENIED", "PENDING"], n_rows, p=[0.6, 0.2, 0.2])

    base_amounts = rng.normal(5000, 1500, n_rows)
    base_amounts = np.clip(base_amounts, 500, 20000)

    outlier_idx = rng.choice(n_rows, int(0.02 * n_rows), replace=False)
    base_amounts[outlier_idx] *= 5

    start = datetime(2023, 1, 1)
    dates = [start + timedelta(days=int(x)) for x in rng.integers(0, 365, n_rows)]

    zip_codes = rng.integers(10000, 99999, n_rows).astype(str)

    years = rng.integers(1940, 2005, n_rows)
    months = rng.integers(1, 13, n_rows)
    days = rng.integers(1, 28, n_rows)
    dobs = [datetime(int(y), int(m), int(d)).strftime("%Y-%m-%d") for y, m, d in zip(years, months, days)]

    df = pd.DataFrame(
        {
            "claim_id": claim_ids,
            "patient_id": patient_ids,
            "doctor_id": doctor_ids,
            "diagnosis_code": diagnosis_codes,
            "procedure_code": procedure_codes,
            "claim_amount": base_amounts.round(2),
            "claim_date": dates,
            "status": statuses,
            "zip_code": zip_codes,
            "dob": dobs,
        }
    )

    n_dup = int(dup_fraction * n_rows)
    if n_dup > 0:
        dup_sample = df.sample(n_dup, random_state=random_seed)
        df = pd.concat([df, dup_sample], ignore_index=True)

    n_missing = int(missing_fraction * df.size)
    for _ in range(n_missing):
        r = rng.integers(0, df.shape[0])
        c = rng.integers(0, df.shape[1])
        df.iat[r, c] = np.nan

    n_invalid = int(invalid_fraction * df.shape[0])
    invalid_idx = rng.choice(df.index, n_invalid, replace=False)

    for idx in invalid_idx:
        t = rng.choice(["zip", "dob", "amount"])
        if t == "zip":
            df.at[idx, "zip_code"] = rng.choice(["ABC", "123", "999999", "ZIP??"])
        elif t == "dob":
            df.at[idx, "dob"] = rng.choice(["2025-01-01", "1900-13-40", "not-a-date"])
        else:
            df.at[idx, "claim_amount"] = rng.choice([0, -100, -500])

    return df


def basic_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = [
        "claim_id",
        "patient_id",
        "doctor_id",
        "diagnosis_code",
        "procedure_code",
        "claim_amount",
        "claim_date",
        "status",
        "zip_code",
        "dob",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan

    df["claim_date"] = pd.to_datetime(df["claim_date"], errors="coerce")
    df["claim_amount"] = pd.to_numeric(df["claim_amount"], errors="coerce")
    return df


def run_etl() -> Tuple[pd.DataFrame, pd.DataFrame]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = generate_mock_claims_data()
    raw_path = OUTPUT_DIR / "claims_raw.csv"
    raw_df.to_csv(raw_path, index=False)

    processed_df = basic_transform(raw_df)
    processed_path = OUTPUT_DIR / "claims_processed.csv"
    processed_df.to_csv(processed_path, index=False)

    return raw_df, processed_df
