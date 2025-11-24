"""
Exploratory Data Analysis (EDA) module for diabetes readmission prediction.
Contains functions for visualizing and analyzing the data.
"""

from typing import List

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

from . import config


def setup_altair():
    """Configure Altair for optimal performance."""
    alt.data_transformers.enable("vegafusion")


def get_continuous_features() -> List[str]:
    """Get list of continuous features (excluding age_mid)."""
    return [f for f in config.CONTINUOUS_FEATURES if f != "age_mid"]


def plot_distributions_altair_with_kde(df: pd.DataFrame, numeric_cols):
    """
    Generates histograms with a KDE overlay and box plots side-by-side.
    """
    charts = []
    for col in numeric_cols:
        is_discrete_int = (
            pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() < 30
        )
        x_binning = alt.Bin(step=1) if is_discrete_int else alt.Bin(maxbins=30)

        base = alt.Chart(df).properties(height=200)

        hist = (
            base.mark_bar(opacity=0.6)
            .encode(
                alt.X(
                    f"{col}:Q",
                    bin=x_binning,
                    title=col,
                    axis=alt.Axis(format="d", labelAngle=0),
                ),
                alt.Y("count():Q", title="Count", axis=alt.Axis(titleColor="#1f77b4")),
            )
            .properties(width=350)
        )

        density = (
            base.transform_density(col, as_=[col, "density"])
            .mark_line(color="orange", strokeWidth=3)
            .encode(
                x=f"{col}:Q",
                y=alt.Y(
                    "density:Q", axis=alt.Axis(title="Density", titleColor="orange")
                ),
            )
        )

        distribution_plot = (
            alt.layer(hist, density)
            .resolve_scale(y="independent")
            .properties(title=f"Distribution of {col}")
        )

        boxplot = (
            alt.Chart(df)
            .mark_boxplot()
            .encode(
                x=alt.X("readmitted:N", title="Readmitted"),
                y=alt.Y(f"{col}:Q", title=col),
            )
            .properties(title=f"{col} by Readmission", width=200, height=200)
        )

        combined_chart = distribution_plot | boxplot
        charts.append(combined_chart)

    final_chart = alt.vconcat(*charts)

    return final_chart


def create_box_plots(
    df: pd.DataFrame,
    features: List[str] = None,
    target_col: str = "readmitted",
    ncols: int = 3,
) -> alt.Chart:
    """
    Create box plots for continuous features by target class.

    Args:
        df: DataFrame containing features and target
        features: List of feature names to plot
        target_col: Name of the target column
        ncols: Number of columns in the grid

    Returns:
        Altair chart with box plots
    """
    if features is None:
        features = get_continuous_features()

    box_plots = []
    for feature in features:
        chart = (
            alt.Chart(df)
            .mark_boxplot()
            .encode(
                x=alt.X(f"{target_col}:N", title="Readmitted Class"),
                y=alt.Y(f"{feature}:Q", title=feature.replace("_", " ").title()),
                color=alt.Color(f"{target_col}:N", title="Readmitted Class").scale(
                    scheme=config.COLOR_SCHEME
                ),
                tooltip=[target_col, feature],
            )
            .properties(
                title=f"Box Plot of {feature.replace('_', ' ').title()} by Readmission",
                width=config.PLOT_WIDTH,
                height=config.PLOT_HEIGHT,
            )
        )
        box_plots.append(chart)

    # Create grid layout
    rows = [
        alt.hconcat(*box_plots[i : i + ncols]) for i in range(0, len(box_plots), ncols)
    ]
    return alt.vconcat(*rows)


def create_violin_plots(
    df: pd.DataFrame,
    features: List[str] = None,
    target_col: str = "readmitted",
    ncols: int = 4,
) -> alt.Chart:
    """
    Create violin plots for continuous features by target class.

    Args:
        df: DataFrame containing features and target (should use raw, unstandardized data)
        features: List of feature names to plot
        target_col: Name of the target column
        ncols: Number of columns in the grid

    Returns:
        Altair chart with violin plots
    """
    if features is None:
        features = get_continuous_features()

    violin_plots = []
    for feature in features:
        chart = (
            alt.Chart(df)
            .transform_density(
                density=feature, as_=[feature, "density"], groupby=[target_col]
            )
            .mark_area(orient="horizontal")
            .encode(
                y=alt.Y(f"{feature}:Q", title=feature.replace("_", " ").title()),
                x=alt.X(
                    "density:Q",
                    stack="center",
                    impute=None,
                    title=None,
                    axis=alt.Axis(labels=False, values=[0], grid=False, ticks=True),
                ),
                color=alt.Color(
                    f"{target_col}:N", legend=alt.Legend(title="Readmitted")
                ).scale(scheme="tableau20"),
            )
            .properties(
                title=f"Distribution of {feature.replace('_', ' ').title()} by Readmission",
                width=config.PLOT_WIDTH,
                height=250,
            )
        )
        violin_plots.append(chart)

    # Create grid layout
    rows = [
        alt.hconcat(*violin_plots[i : i + ncols])
        for i in range(0, len(violin_plots), ncols)
    ]
    return alt.vconcat(*rows)


def create_correlation_heatmap(
    df: pd.DataFrame, features: List[str] = None, figsize: tuple = (12, 10)
) -> plt.Figure:
    """
    Create a correlation heatmap for numerical features.

    Args:
        df: DataFrame containing the features
        features: List of feature names to include (if None, uses all numeric columns)
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure
    """
    if features is not None:
        df_corr = df[features].corr()
    else:
        df_corr = df.select_dtypes(include=[np.number]).corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df_corr,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap of Features", fontsize=16, pad=20)
    plt.tight_layout()
    return fig


def create_target_distribution(
    Y: pd.Series, target_name: str = "readmitted"
) -> alt.Chart:
    """
    Create a bar plot showing the distribution of the target variable.

    Args:
        Y: Target variable Series
        target_name: Name of the target variable

    Returns:
        Altair chart
    """
    chart = (
        alt.Chart(Y.reset_index())
        .mark_bar()
        .encode(
            x=alt.X(f"{Y.name}:N", title=target_name.replace("_", " ").title()),
            y=alt.Y("count()", title="Count"),
            color=alt.Color(f"{Y.name}:N", legend=None).scale(scheme="tableau20"),
        )
        .properties(
            title=f"Distribution of {target_name.replace('_', ' ').title()}",
            width=config.PLOT_WIDTH,
            height=300,
        )
    )
    return chart


def print_data_summary(df: pd.DataFrame):
    """
    Print a comprehensive summary of the dataset.

    Args:
        df: DataFrame to summarize
    """
    print("=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of Features: {len(df.columns)}")
    print(f"Number of Samples: {len(df)}")

    print("\n" + "-" * 70)
    print("Missing Values:")
    print("-" * 70)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values")
    else:
        print(missing[missing > 0])

    print("\n" + "-" * 70)
    print("Data Types:")
    print("-" * 70)
    print(df.dtypes.value_counts())

    print("\n" + "-" * 70)
    print("Sample Data (first 5 rows):")
    print("-" * 70)
    print(df.head())

    print("\n" + "=" * 70)


def generate_eda_report(
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    raw_df: pd.DataFrame = None,
    save_plots: bool = False,
    output_dir: str = None,
):
    """
    Generate a comprehensive EDA report with all visualizations.

    Args:
        X_train: Training features
        Y_train: Training target
        raw_df: Raw DataFrame for violin plots (with original scales)
        save_plots: Whether to save plots to files
        output_dir: Directory to save plots (uses config.RESULTS_DIR if None)
    """
    import os

    if output_dir is None:
        output_dir = config.RESULTS_DIR

    print("\n" + "=" * 70)
    print("GENERATING EDA REPORT")
    print("=" * 70)

    # Combine train data for visualization
    train_combined = X_train.copy()
    train_combined["readmitted"] = Y_train.reset_index(drop=True)

    # Print summary
    print("\n1. Data Summary")
    print_data_summary(train_combined)

    # Target distribution
    print("\n2. Target Distribution")
    fig_target = create_target_distribution(Y_train)
    if save_plots:
        fig_target.savefig(
            os.path.join(output_dir, "target_distribution.png"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()

    # Continuous features distribution
    print("\n3. Feature Distributions (Histograms)")
    continuous_features = get_continuous_features()
    fig_hist = plot_distributions_altair_with_kde(train_combined, continuous_features)
    if save_plots:
        fig_hist.save(os.path.join(output_dir, "feature_distributions.html"))
    display(fig_hist)

    # Correlation heatmap
    print("\n4. Feature Correlations")
    fig_corr = create_correlation_heatmap(train_combined)
    if save_plots:
        fig_corr.savefig(
            os.path.join(output_dir, "correlation_heatmap.png"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()

    # KDE plots (skipped - function not implemented)
    print("\n5. KDE Plots by Target Class (skipped - function not implemented)")

    # Box plots
    print("\n6. Box Plots by Target Class")
    box_chart = create_box_plots(train_combined)
    if save_plots:
        box_chart.save(os.path.join(output_dir, "box_plots.html"))
    display(box_chart)

    # Violin plots (use raw data if available)
    if raw_df is not None:
        print("\n7. Violin Plots by Target Class (using raw data)")
        violin_chart = create_violin_plots(raw_df)
        if save_plots:
            violin_chart.save(os.path.join(output_dir, "violin_plots.html"))
        display(violin_chart)

    print("\n" + "=" * 70)
    print("EDA REPORT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print("EDA module loaded successfully")
    print("Available functions:")
    print("  - create_distribution_histograms")
    print("  - create_kde_plots")
    print("  - create_box_plots")
    print("  - create_violin_plots")
    print("  - create_correlation_heatmap")
    print("  - create_target_distribution")
    print("  - generate_eda_report")
