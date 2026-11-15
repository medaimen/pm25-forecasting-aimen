"""Data cleaning and train-only missing-value preprocessing."""

from collections.abc import Sequence

import pandas as pd
from sklearn.impute import SimpleImputer

from src.config import MISSING_COLUMNS, TARGET_COLUMN


def clean_pm25(data: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, remove invalid PM2.5 rows, and sort observations."""
    required_columns = {"date", TARGET_COLUMN}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    cleaned = data.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="raise")
    if "datetime_utc" in cleaned.columns:
        cleaned["datetime_utc"] = pd.to_datetime(
            cleaned["datetime_utc"],
            errors="raise",
        )

    valid_target = cleaned[TARGET_COLUMN].gt(0) & cleaned[TARGET_COLUMN].lt(500)
    cleaned = cleaned.loc[valid_target].copy()
    if "year" not in cleaned.columns:
        cleaned["year"] = cleaned["date"].dt.year

    sort_columns = ["date"]
    if "sensor_name" in cleaned.columns:
        sort_columns.append("sensor_name")
    return cleaned.sort_values(sort_columns).reset_index(drop=True)


def chronological_train_test_split(
    data: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split at a unique-date boundary so train dates precede test dates."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if data.empty:
        raise ValueError("Cannot split an empty dataset.")

    dates = pd.to_datetime(data["date"])
    unique_dates = dates.drop_duplicates().sort_values()
    split_index = int(len(unique_dates) * (1 - test_fraction))
    if split_index <= 0 or split_index >= len(unique_dates):
        raise ValueError("The date range is too small for the requested split.")

    split_date = unique_dates.iloc[split_index]
    train = data.loc[dates < split_date].copy()
    test = data.loc[dates >= split_date].copy()
    if train.empty or test.empty or train["date"].max() >= test["date"].min():
        raise ValueError("The chronological split did not produce disjoint date ranges.")
    return train, test, split_date


def add_missingness_indicators(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    columns: Sequence[str] = MISSING_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add missingness flags without learning any values from the test set."""
    train = train_features.copy()
    test = test_features.copy()
    for column in columns:
        if column not in train or column not in test:
            raise ValueError(f"Missingness column is absent from train or test: {column}")
        train[f"{column}_missing"] = train[column].isna().astype(int)
        test[f"{column}_missing"] = test[column].isna().astype(int)
    return train, test


def impute_from_training_data(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    columns: Sequence[str] = MISSING_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer]:
    """Fit median imputations on training rows and apply them to test rows."""
    train = train_features.copy()
    test = test_features.copy()
    selected_columns = list(columns)
    if not selected_columns:
        raise ValueError("At least one imputation column must be provided.")
    if any(column not in train or column not in test for column in selected_columns):
        raise ValueError("Every imputation column must exist in both data frames.")

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train[selected_columns] = imputer.fit_transform(train[selected_columns])
    test[selected_columns] = imputer.transform(test[selected_columns])
    return train, test, imputer
