"""Shared project paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
EXPLORATION_RESULTS_DIR = RESULTS_DIR / "exploration"
FIGURES_DIR = PROJECT_ROOT / "figures"
FOLDS_DIR = PROJECT_ROOT / "folds"

COMBINED_DATA_PATH = RAW_DATA_DIR / "combined_gee_pm25.csv"

SEED = 42
TEST_FRACTION = 0.2
N_SPLITS = 5
TARGET_COLUMN = "pm25"
MISSING_COLUMNS = ("NO2", "CO", "O3")
