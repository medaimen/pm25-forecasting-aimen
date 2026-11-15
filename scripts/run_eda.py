"""Generate core EDA tables, figures, and reproducible fold-index files."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller, pacf

from src.config import (
    COMBINED_DATA_PATH,
    FIGURES_DIR,
    FOLDS_DIR,
    N_SPLITS,
    RESULTS_DIR,
    SEED,
    TARGET_COLUMN,
)
from src.preprocessing import chronological_train_test_split, clean_pm25
from src.splits import (
    random_fold_indices,
    save_folds_json,
    spatial_fold_indices,
    temporal_fold_indices,
)


def save_exploratory_tables(data: pd.DataFrame) -> None:
    """Write descriptive and temporal statistics to results/."""
    numeric_data = data.select_dtypes(include="number").drop(
        columns=["id"],
        errors="ignore",
    )
    descriptive = numeric_data.describe().T
    descriptive["missing_values"] = data[numeric_data.columns].isna().sum()
    descriptive["missing_percentage"] = (
        data[numeric_data.columns].isna().mean() * 100
    )
    descriptive.to_csv(RESULTS_DIR / "Exploratory_Data.csv")

    daily_pm25 = (
        data.groupby("date")[TARGET_COLUMN]
        .mean()
        .sort_index()
        .asfreq("D")
    )
    daily_interpolated = daily_pm25.interpolate(
        method="time",
        limit_area="inside",
    ).dropna()
    adf_result = adfuller(daily_interpolated, autolag="AIC")
    ljung_box = acorr_ljungbox(
        daily_interpolated,
        lags=[7, 14, 30],
        return_df=True,
    )
    acf_values = acf(daily_interpolated, nlags=30, fft=False)
    pacf_values = pacf(daily_interpolated, nlags=30, method="ywm")

    temporal_rows = [
        {"Metric": "ADF statistic", "Value": adf_result[0]},
        {"Metric": "ADF p-value", "Value": adf_result[1]},
        {"Metric": "ADF used lags", "Value": adf_result[2]},
        {"Metric": "ADF observations", "Value": adf_result[3]},
    ]
    for lag in ljung_box.index:
        temporal_rows.append({
            "Metric": f"Ljung-Box statistic (lag {lag})",
            "Value": ljung_box.loc[lag, "lb_stat"],
        })
        temporal_rows.append({
            "Metric": f"Ljung-Box p-value (lag {lag})",
            "Value": ljung_box.loc[lag, "lb_pvalue"],
        })
    for lag in range(1, 31):
        temporal_rows.append({"Metric": f"ACF lag {lag}", "Value": acf_values[lag]})
        temporal_rows.append({"Metric": f"PACF lag {lag}", "Value": pacf_values[lag]})

    pd.DataFrame(temporal_rows).to_csv(
        RESULTS_DIR / "Exploratory_temporal.csv",
        index=False,
    )


def save_exploratory_figures(data: pd.DataFrame) -> None:
    """Save the main distribution and correlation figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["pm25", "AOD", "temperature_celsius"]
    fig, axes = plt.subplots(1, len(columns), figsize=(15, 4))
    for axis, column in zip(axes, columns):
        sns.histplot(data=data, x=column, bins=40, ax=axis)
        axis.set_title(column)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "eda_histogrammes.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    numeric_data = data.select_dtypes(include="number").drop(
        columns=["id"],
        errors="ignore",
    )
    fig, axis = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        numeric_data.corr(),
        cmap="coolwarm",
        center=0,
        ax=axis,
    )
    axis.set_title("Correlation Matrix")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "correlation_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    daily_pm25 = (
        data.groupby("date")[TARGET_COLUMN]
        .mean()
        .sort_index()
        .asfreq("D")
        .interpolate(method="time", limit_area="inside")
        .dropna()
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(daily_pm25, lags=30, ax=axes[0])
    plot_pacf(daily_pm25, lags=30, method="ywm", ax=axes[1])
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "acf_pacf_timeseries.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_validation_folds(data: pd.DataFrame) -> None:
    """Save five chronological, station-grouped, and random CV folds."""
    train_data, _, _ = chronological_train_test_split(data)
    save_folds_json(
        temporal_fold_indices(train_data["date"], N_SPLITS, gap=1),
        FOLDS_DIR / "folds_temporal.json",
    )
    save_folds_json(
        spatial_fold_indices(train_data["id"], N_SPLITS),
        FOLDS_DIR / "folds_spatial.json",
    )
    save_folds_json(
        random_fold_indices(len(train_data), N_SPLITS, SEED),
        FOLDS_DIR / "folds_standard.json",
    )


def main() -> None:
    """Run the current EDA milestone without fitting a predictive model."""
    raw_data = pd.read_csv(COMBINED_DATA_PATH)
    clean_data = clean_pm25(raw_data)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FOLDS_DIR.mkdir(parents=True, exist_ok=True)
    save_exploratory_tables(clean_data)
    save_exploratory_figures(clean_data)
    save_validation_folds(clean_data)

    print("EDA tables saved in:", RESULTS_DIR)
    print("Figures saved in:", FIGURES_DIR)
    print("Fold indices saved in:", FOLDS_DIR)


if __name__ == "__main__":
    main()
