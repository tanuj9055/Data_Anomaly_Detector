from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

N_ROWS = 2000
RANDOM_SEED = 42

IFOREST_CONTAMINATION = 0.03
IFOREST_ESTIMATORS = 100
