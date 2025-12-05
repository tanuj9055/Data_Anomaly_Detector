import logging

from config import OUTPUT_DIR
from etl import run_etl
from anomaly_detector import DataQualityAnomalyDetector


def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def run_pipeline():
    logger = logging.getLogger("pipeline")

    raw_df, processed_df = run_etl()
    detector = DataQualityAnomalyDetector(processed_df)

    dup = detector.find_duplicates()
    dup.to_csv(OUTPUT_DIR / "duplicates.csv", index=False)

    miss = detector.find_missing_values()
    miss.to_csv(OUTPUT_DIR / "missing_values.csv", index=False)

    invalid = detector.find_invalid_formats()
    invalid.to_csv(OUTPUT_DIR / "invalid_formats.csv", index=False)

    outliers = detector.find_outliers_iqr()
    outliers.to_csv(OUTPUT_DIR / "outliers_iqr.csv", index=False)

    ml_anom = detector.find_anomalies_isolation_forest()
    ml_anom.to_csv(OUTPUT_DIR / "anomalies_isolation_forest.csv", index=False)

    summary = detector.summary()
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    logger.info("Summary:\n%s", summary)

    sev = detector.severity_report()
    sev.to_csv(OUTPUT_DIR / "severity_report.csv", index=False)
    logger.info("Severity:\n%s", sev)


if __name__ == "__main__":
    setup_logging()
    run_pipeline()
