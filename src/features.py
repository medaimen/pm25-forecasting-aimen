"""Leakage-aware calendar, lag, and rolling features."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def add_cyclic_calendar_features(
    data: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    """Add sine/cosine encodings for month and day of week."""
    result = data.copy()
    dates = pd.to_datetime(result[date_column], errors="raise")

    month_angle = 2 * np.pi * (dates.dt.month - 1) / 12
    weekday_angle = 2 * np.pi * dates.dt.dayofweek / 7
    result["month_sin"] = np.sin(month_angle)
    result["month_cos"] = np.cos(month_angle)
    result["day_of_week_sin"] = np.sin(weekday_angle)
    result["day_of_week_cos"] = np.cos(weekday_angle)
    return result


def add_lag_features(
    data: pd.DataFrame,
    columns: Sequence[str],
    lags: Sequence[int],
    group_column: str,
    date_column: str = "date",
) -> pd.DataFrame:
    """Add within-group lag values after sorting by date."""
    if any(lag < 1 for lag in lags):
        raise ValueError("Lags must be positive so features use past observations.")
    result = data.sort_values([group_column, date_column]).copy()
    grouped = result.groupby(group_column, sort=False)
    for column in columns:
        if column not in result.columns:
            raise ValueError(f"Unknown lag source column: {column}")
        for lag in lags:
            result[f"{column}_lag_{lag}"] = grouped[column].shift(lag)
    return result


def add_rolling_mean_features(
    data: pd.DataFrame,
    columns: Sequence[str],
    windows: Sequence[int],
    group_column: str,
    date_column: str = "date",
) -> pd.DataFrame:
    """Add past-only rolling means, excluding the current observation."""
    if any(window < 1 for window in windows):
        raise ValueError("Rolling windows must be positive.")
    result = data.sort_values([group_column, date_column]).copy()
    for column in columns:
        if column not in result.columns:
            raise ValueError(f"Unknown rolling source column: {column}")
        grouped = result.groupby(group_column, sort=False)[column]
        for window in windows:
            name = f"{column}_rolling_mean_{window}"
            result[name] = grouped.transform(
                lambda values: values.shift(1).rolling(
                    window=window,
                    min_periods=window,
                ).mean()
            )
    return result
